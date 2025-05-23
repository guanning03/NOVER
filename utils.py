import re
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
import wandb
import numpy as np

def extract_content(text: str, tag: str) -> str:
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_all_content(text: str, tag: str) -> list:
    pattern = f"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]

def selective_log_softmax(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, vocab_size = logits.shape
    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)
    
    log_probs = F.log_softmax(flat_logits, dim=-1)
    target_log_probs = log_probs[torch.arange(flat_targets.size(0)), flat_targets]
    return target_log_probs.reshape(batch_size, seq_len)

def calculate_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    question: str = "",
    reasoning: str = "",
    target: str = "",
    use_natural_format: bool = False,
    use_tagged_format: bool = True
) -> float:
    if use_tagged_format:
        prompt = f"""
Question: {question}

Answer the question and return in the following format:

<think>
...
</think>

<answer>
...
</answer>
"""
        if reasoning:
            prompt = prompt + f"\n<think>\n{reasoning}\n</think>\n"
            
        target = f"\n<answer>\n{target}\n</answer>"
        
        combined = prompt + target
    elif use_natural_format:
        if reasoning:
            prompt = f"For the question: {question}, I think {reasoning}, so the answer is "
        else:
            prompt = f"For the question: {question}, the answer is "
        combined = prompt + target
    else:
        prompt = question + reasoning
        combined = prompt + target

    inputs = tokenizer(combined, return_tensors="pt").to(model.device)
    input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
    
    prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_len = prompt_tokens.size(-1)

    reasoning_tokens = tokenizer(reasoning, return_tensors="pt").input_ids.to(model.device)
    reasoning_len = reasoning_tokens.size(-1)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        
        targets = input_ids[:, 1:]
        
        target_logits = logits[:, prompt_len-1:, :]
        target_ids = targets[:, prompt_len-1:]
        target_mask = attention_mask[:, 1:][:, prompt_len-1:]
        
        log_probs = selective_log_softmax(target_logits, target_ids)
        
        log_probs = log_probs * target_mask
        nll = -log_probs.sum() / target_mask.sum()

        norm_term = max(1, 1 + np.log10(reasoning_len))
        raw_nll = nll
        nll = nll / norm_term
        
    return torch.exp(nll).item(), torch.exp(raw_nll).item()

def safe_wandb_log(data, step=None):
    if wandb.run is None:
        return
    
    fixed_data = {}
    for key, value in data.items():
        if isinstance(key, str) and key.startswith("metrics/") and isinstance(value, str):
            fixed_key = key.replace("metrics/", "text_metrics/")
            fixed_data[fixed_key] = value
        else:
            fixed_data[key] = value
    
    if step is None:
        try:
            current_step = wandb.run.history._step
            step = current_step
        except:
            step = 0
    
    wandb.log(fixed_data)