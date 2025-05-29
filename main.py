import wandb
import torch
from config import (
    setup_environment, init_wandb, get_training_config, PEFT_CONFIG,
    MODEL_NAME, DATASET_NAME, RESUME_FROM_CHECKPOINT, VALIDATION_SIZE,
    save_config_to_yaml, INTERMEDIATE_TAG, FINAL_TAG
)
from data_loader import load_train_val_dataset
from trainer import CustomGRPOTrainer
from datetime import timedelta
from reward_functions import tag_format_reward, reasoning_reward, efficiency_reward
import argparse

torch.distributed.init_process_group(backend='nccl', timeout=timedelta(seconds=600))

def main():
    parser = argparse.ArgumentParser(description="NOVER Training")
    parser.add_argument("--intermediate-tag", type=str, default=INTERMEDIATE_TAG,
                        help=f"Custom intermediate tag (default: {INTERMEDIATE_TAG})")
    parser.add_argument("--final-tag", type=str, default=FINAL_TAG,
                        help=f"Custom final tag (default: {FINAL_TAG})")
    args = parser.parse_args()
    
    # Override config values if provided via command line
    intermediate_tag = args.intermediate_tag
    final_tag = args.final_tag
    
    print(f"Using intermediate tag: <{intermediate_tag}>")
    print(f"Using final tag: <{final_tag}>")
    
    setup_environment()
    
    init_wandb()
    
    save_config_to_yaml()
    
    train_data, val_data = load_train_val_dataset(dataset_name=DATASET_NAME, val_size=VALIDATION_SIZE)
    
    training_config = get_training_config()
    
    # Create partial functions with custom tags
    def tag_format_reward_custom(completions, **kwargs):
        return tag_format_reward(completions, intermediate_tag=intermediate_tag, final_tag=final_tag, **kwargs)
    
    def reasoning_reward_custom(completions, **kwargs):
        # Pass through to reasoning_reward with custom tags
        kwargs.update({"intermediate_tag": intermediate_tag, "final_tag": final_tag})
        return reasoning_reward(completions, **kwargs)
    
    def efficiency_reward_custom(completions, **kwargs):
        # Pass through to efficiency_reward with custom tags
        kwargs.update({"intermediate_tag": intermediate_tag, "final_tag": final_tag})
        return efficiency_reward(completions, **kwargs)
    
    reward_funcs = [
        tag_format_reward_custom,
        reasoning_reward_custom,
        efficiency_reward_custom
    ]
    
    trainer = CustomGRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=reward_funcs,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_config,
        peft_config=PEFT_CONFIG,
        custom_tags={
            "intermediate_tag": intermediate_tag,
            "final_tag": final_tag
        }
    )
    
    trainer.train(resume_from_checkpoint=MODEL_NAME if RESUME_FROM_CHECKPOINT else None)
    wandb.finish()
    

if __name__ == "__main__":
    main()