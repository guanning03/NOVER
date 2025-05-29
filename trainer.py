from trl import GRPOTrainer
import torch
import wandb
import warnings
from torch import nn
from accelerate.utils import gather, gather_object, broadcast_object_list
from trl.data_utils import is_conversational, maybe_apply_chat_template, apply_chat_template
from trl.models import unwrap_model_for_generation
from trl.trainer.utils import pad
from reward_functions import set_reference_model, validation_accuracy
from typing import Union, Any, List
from transformers import Trainer
from trl.extras.profiling import profiling_context, profiling_decorator
from utils import safe_wandb_log
from trl.trainer.callbacks import SyncRefModelCallback
import copy
import Levenshtein
import re
import statistics
import numpy as np
import pandas as pd
import random
from trl.trainer.callbacks import TrainerCallback
from peft import get_peft_model_state_dict, set_peft_model_state_dict
import os
import json
from datetime import datetime
class SyncRefLoraModelCallback(TrainerCallback):
    def __init__(self, ref_model, accelerator, policy_model):
        self.accelerator = accelerator
        self.ref_model = ref_model
        self.policy_model = policy_model
        
    def on_save(self, args, state, control, **kwargs):
        try:
            policy_adapter_state = get_peft_model_state_dict(self.policy_model)
            ref_adapter_state = get_peft_model_state_dict(self.ref_model)
            common_keys = set(ref_adapter_state.keys()).intersection(set(policy_adapter_state.keys()))
            missing_keys = set(ref_adapter_state.keys()) - set(policy_adapter_state.keys())
            extra_keys = set(policy_adapter_state.keys()) - set(ref_adapter_state.keys())
            if missing_keys:
                print(f"[DEBUG] First few missing keys: {list(missing_keys)[:3]}")
            if extra_keys:
                print(f"[DEBUG] First few extra keys: {list(extra_keys)[:3]}")
            alpha = args.ref_model_mixup_alpha

            mixed_adapter_state_dict = {}
            for key in policy_adapter_state:
                mixed_adapter_state_dict[key] = alpha * ref_adapter_state[key] + (1 - alpha) * policy_adapter_state[key]

            set_peft_model_state_dict(self.ref_model, mixed_adapter_state_dict)

            print(f"[INFO] Successfully synced adapter from policy model")
            
        except Exception as e:
            print(f"[ERROR] Failed to sync adapter: {e}")
            import traceback
            print(traceback.format_exc())


class CustomGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, custom_tags=None, **kwargs):
        self.custom_tags = custom_tags or {}
        self.intermediate_tag = self.custom_tags.get("intermediate_tag", "think")
        self.final_tag = self.custom_tags.get("final_tag", "answer")
        
        super().__init__(*args, **kwargs)

        self.remove_callback(SyncRefModelCallback)
        self._setup_reference_model()
        
        set_reference_model(self.ref_model, self.processing_class)
        
        self.current_batch = None
        
        self.diversity_history = []
        self.diversity_steps = []
        
        if self.args.sync_ref_model:
            if hasattr(self.model, 'get_adapter_state_dict'):
                self.add_callback(SyncRefLoraModelCallback(ref_model=self.ref_model, accelerator=self.accelerator, policy_model=self.model))
        
        self.validation_results_dir = os.path.join(self.args.output_dir, "validation_results")
        os.makedirs(self.validation_results_dir, exist_ok=True)
        
    def _setup_reference_model(self):
        
        if not self.args.sync_ref_model:
            print(f"[INFO] No ref sync, ref = base model")
            self.ref_model = self.model.get_base_model()
            set_reference_model(self.ref_model, self.processing_class)
            return
            
        try:
            self.ref_model = copy.deepcopy(self.model)
            
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
            
            set_reference_model(self.ref_model, self.processing_class)
            
        except Exception as e:
            import traceback
            print(traceback.format_exc())

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1) 

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        advantages = inputs["advantages"]
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        if self.accelerator.is_main_process and self.state.global_step % self.args.logging_steps == 0:
            policy_losses1 = (per_token_loss1 * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
            policy_losses2 = (per_token_loss2 * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
            
            clip_percentage = (policy_losses1 > policy_losses2).float().mean().item() * 100
            
            print(f"\nLoss Analysis (Step {self.state.global_step}):")
            if self.beta != 0.0:
                print(f"KL term (β={self.beta}): {self._metrics[mode]['kl'][-1]:.6f}")
            print(f"Final loss: {loss.item():.6f}")
            print(f"Clip percentage: {clip_percentage:.2f}% of tokens were clipped")
            
            if abs(loss.item()) < 1e-5:
                print("WARNING: Loss is near zero - policy and KL terms may be canceling out")
            if clip_percentage < 1.0:
                print("NOTE: Very low clip percentage - consider adjusting epsilon parameters")
            elif clip_percentage > 80.0:
                print("NOTE: Very high clip percentage - consider adjusting epsilon parameters")
            print("-"*50 + "\n")

        return loss


    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        reference_text = [maybe_apply_chat_template(example, self.processing_class)["reference"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if self.args.use_vllm:
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                with profiling_context(self, "vLLM.generate"):
                    completion_ids = self.vllm_client.generate(
                        prompts=ordered_set_of_prompts,
                        n=self.num_generations,
                        repetition_penalty=self.repetition_penalty,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=-1 if self.top_k is None else self.top_k,
                        min_p=0.0 if self.min_p is None else self.min_p,
                        max_tokens=self.max_completion_length,
                        guided_decoding_regex=self.guided_decoding_regex,
                    )
            else:
                completion_ids = [None] * len(all_prompts_text)
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)

        if is_eos.size(1) > 0:
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]

        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module): 
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):
                if isinstance(
                    reward_func, nn.Module
                ): 
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    
                    reward_kwargs["global_step"] = self.state.global_step
                    
                    output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                    
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        rewards_per_func = gather(rewards_per_func)
    
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.args.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        advantages_full = advantages.clone()

        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "eval" and reference_text:
            try:
                print(f"\n[INFO] Running validation accuracy calculation at step {self.state.global_step}")
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                val_kwargs = {key: [example[key] for example in inputs] for key in keys}
                
                val_kwargs["global_step"] = self.state.global_step
                
                if 'reference' in val_kwargs:
                    del val_kwargs['reference']
                
                accuracy_scores = validation_accuracy(
                    prompts=prompts,
                    completions=completions,
                    reference=reference_text,
                    **val_kwargs
                )
                
                valid_scores = [score for score in accuracy_scores if score is not None]
                overall_accuracy = sum(valid_scores) / max(1, len(valid_scores))
                
                self._metrics[mode]["validation/edit_distance_similarity"].append(overall_accuracy)
                print(f"[INFO] Validation edit distance similarity: {overall_accuracy:.4f}")
                
                if self.num_generations > 1:
                    grouped_scores = np.array(accuracy_scores).reshape(-1, self.num_generations)
                    
                    max_similarity_at_k = [float(max(group)) for group in grouped_scores]
                    
                    overall_max_similarity_at_k = sum(max_similarity_at_k) / max(1, len(max_similarity_at_k))
                    
                    self._metrics[mode][f"validation/max_similarity@{self.num_generations}"].append(overall_max_similarity_at_k)
                    print(f"[INFO] Validation max similarity@{self.num_generations}: {overall_max_similarity_at_k:.4f}")
                
                if self.accelerator.is_main_process:
                    prompts_to_save = gather_object(prompts_text)
                    completions_to_save = gather_object(completions_text)
                    reference_to_save = gather_object(reference_text)
                    accuracy_scores_to_save = gather_object(accuracy_scores)
                    rewards_to_save = rewards.tolist()
                    
                    reward_func_names = []
                    reward_raw_values = {}
                    
                    for i, reward_func in enumerate(self.reward_funcs):
                        if isinstance(reward_func, nn.Module):
                            reward_func_name = reward_func.config._name_or_path.split("/")[-1]
                        else:
                            reward_func_name = reward_func.__name__
                        
                        reward_func_names.append(reward_func_name)
                        reward_raw_values[reward_func_name] = rewards_per_func[:, i].tolist()
                    
                    validation_results = []
                    for i, (prompt, completion, reference, accuracy) in enumerate(
                        zip(prompts_to_save, completions_to_save, reference_to_save, accuracy_scores_to_save)
                    ):
                        result = {
                            "id": i,
                            "prompt": prompt,
                            "completion": completion,
                            "reference": reference,
                            "edit_distance_similarity": float(accuracy) if accuracy is not None else None,
                            "total_reward": rewards_to_save[i] if i < len(rewards_to_save) else None,
                            "rewards": {}
                        }
                        
                        for reward_name in reward_func_names:
                            if i < len(reward_raw_values[reward_name]):
                                raw_value = reward_raw_values[reward_name][i]
                                result["rewards"][reward_name] = float(raw_value) if not torch.isnan(torch.tensor(raw_value)) else None
                        
                        validation_results.append(result)
                    
                    results_object = {
                        "step": self.state.global_step,
                        "timestamp": datetime.now().isoformat(),
                        "metrics": {
                            "overall_edit_distance_similarity": overall_accuracy,
                            "mean_reward": rewards.mean().item() if len(rewards) > 0 else None
                        },
                        "reward_functions": reward_func_names,
                        "samples": validation_results
                    }
                    
                    if self.num_generations > 1:
                        results_object["metrics"][f"max_similarity@{self.num_generations}"] = overall_max_similarity_at_k
                    
                    output_file = os.path.join(
                        self.validation_results_dir, 
                        f"validation_results_step_{self.state.global_step}.json"
                    )
                    
                    try:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(results_object, f, indent=2, ensure_ascii=False)
                        print(f"[INFO] Validation results saved to: {output_file}")
                    except Exception as e:
                        print(f"[ERROR] Failed to save validation results: {e}")
                
            except Exception as e:
                print(f"[ERROR] Error calculating validation accuracy: {e}")
                import traceback
                print(traceback.format_exc())
        
        if self.accelerator.is_main_process and self.state.global_step % self.args.logging_steps == 0:
            print("\n" + "-"*50)
            print(f"Step {self.state.global_step} - Rewards & Advantages")
            print("-"*50)
            
            reward_func_names = []
            for reward_func in self.reward_funcs:
                if isinstance(reward_func, nn.Module):
                    reward_func_names.append(reward_func.config._name_or_path.split("/")[-1])
                else:
                    reward_func_names.append(reward_func.__name__)
            
            rewards_by_group = rewards.view(-1, self.num_generations)
            rewards_per_func_by_group = rewards_per_func.view(-1, self.num_generations, len(self.reward_funcs))
            
            for group_idx in range(rewards_by_group.shape[0]):
                print(f"Group {group_idx+1} | Mean: {mean_grouped_rewards[group_idx*self.num_generations]:.4f} | Std: {std_grouped_rewards[group_idx*self.num_generations]:.4f}")
                print("Gen | " + " | ".join([f"{name}" for name in reward_func_names]) + " | Total | Adv")
                print("-" * 50)
                
                for gen_idx in range(self.num_generations):
                    global_idx = group_idx * self.num_generations + gen_idx
                    
                    reward_components = []
                    for func_idx in range(len(self.reward_funcs)):
                        component_value = rewards_per_func[global_idx, func_idx].item()
                        if torch.isnan(torch.tensor(component_value)):
                            reward_components.append("N/A")
                        else:
                            weight = self.reward_weights[func_idx].item()
                            weighted_value = component_value * weight
                            reward_components.append(f"{weighted_value:.2f}")
                    
                    components_str = " | ".join(reward_components)
                    print(f"{gen_idx+1:3d} | {components_str} | {rewards[global_idx]:.2f} | {advantages[global_idx]:.2f}")
                
                print("")

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        mode = "eval" if self.control.should_evaluate else "train"

        if mode == "train":
            self._total_train_tokens += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            reference_to_log = gather_object(reference_text)
            rewards_to_log = rewards.tolist()
            advantages_to_log = advantages_full.tolist()  
            
            if self.accelerator.is_main_process:
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reference": reference_to_log,
                        "total_reward": rewards_to_log,
                        "advantage": advantages_to_log,
                    }
                    
                    for i, reward_func in enumerate(self.reward_funcs):
                        if isinstance(reward_func, nn.Module):
                            reward_func_name = reward_func.config._name_or_path.split("/")[-1]
                        else:
                            reward_func_name = reward_func.__name__
                        
                        raw_rewards = rewards_per_func[:, i].tolist()
                        weighted_rewards = (rewards_per_func[:, i] * self.reward_weights[i]).tolist()
                        
                        table[f"raw_{reward_func_name}"] = raw_rewards
                        table[f"weighted_{reward_func_name}"] = weighted_rewards
                    
                    df = pd.DataFrame(table)

                    if wandb.run is not None and self.accelerator.is_main_process:
                        safe_wandb_log({"Combined Rewards": wandb.Table(dataframe=df)}, step=self.state.global_step)
                        
                        diversity_score = self._calculate_completion_diversity(completions_to_log)
                        self._metrics[mode]["diversity_score"].append(diversity_score)
                        
                        self.diversity_history.append(diversity_score)
                        self.diversity_steps.append(self.state.global_step)
                        if len(self.diversity_history) > 100:
                            self.diversity_history = self.diversity_history[-100:]
                            self.diversity_steps = self.diversity_steps[-100:]
                        
                        print(f"\n[DEBUG] Step {self.state.global_step} - Advanced Metrics")
                        print(f"Completion Diversity: {diversity_score:.4f}")
                        print("-" * 50)
                        

                        diversity_analysis = []
                        
                        diversity_analysis.append({
                            "step": str(self.state.global_step),
                            "metric_type": "Diversity",
                            "metric_name": "Completion Diversity",
                            "value": diversity_score,
                            "count": len(completions_to_log)
                        })
                        
                        diversity_df = pd.DataFrame(diversity_analysis)
                        safe_wandb_log({"Diversity Analysis": wandb.Table(dataframe=diversity_df)}, 
                                      step=self.state.global_step)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        } 

    def _calculate_completion_diversity(self, completions: List[str]) -> float:
        if len(completions) <= 1:
            return 0.0
        
        max_pairs = 30
        total_possible_pairs = (len(completions) * (len(completions) - 1)) // 2
        
        if total_possible_pairs > max_pairs:
            distances = []
            sampled_pairs = set()
            attempts = 0
            max_attempts = max_pairs * 3
            
            while len(sampled_pairs) < max_pairs and attempts < max_attempts:
                i = random.randint(0, len(completions) - 1)
                j = random.randint(0, len(completions) - 1)
                if i != j and (min(i, j), max(i, j)) not in sampled_pairs:
                    sampled_pairs.add((min(i, j), max(i, j)))
                    
                    distance = Levenshtein.distance(completions[i], completions[j])
                    max_len = max(len(completions[i]), len(completions[j]))
                    if max_len > 0:
                        normalized_distance = distance / max_len
                        distances.append(normalized_distance)
                attempts += 1
        else:
            distances = []
            for i in range(len(completions)):
                for j in range(i+1, len(completions)):
                    distance = Levenshtein.distance(completions[i], completions[j])
                    max_len = max(len(completions[i]), len(completions[j]))
                    if max_len > 0:
                        normalized_distance = distance / max_len
                        distances.append(normalized_distance)
                    
        return statistics.mean(distances) if distances else 0.0