import wandb
import torch
from config import (
    setup_environment, init_wandb, get_training_config, PEFT_CONFIG,
    MODEL_NAME, DATASET_NAME, RESUME_FROM_CHECKPOINT, VALIDATION_SIZE,
    save_config_to_yaml
)
from data_loader import load_train_val_dataset
from trainer import CustomGRPOTrainer
from datetime import timedelta
from reward_functions import tag_format_reward, reasoning_reward, efficiency_reward

torch.distributed.init_process_group(backend='nccl', timeout=timedelta(seconds=600))

def main():
    
    setup_environment()
    
    init_wandb()
    
    save_config_to_yaml()
    
    train_data, val_data = load_train_val_dataset(dataset_name=DATASET_NAME, val_size=VALIDATION_SIZE)
    
    training_config = get_training_config()
    
    reward_funcs = [
        tag_format_reward,
        reasoning_reward,
        efficiency_reward
    ]
    
    trainer = CustomGRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=reward_funcs,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_config,
        peft_config=PEFT_CONFIG
    )
    
    trainer.train(resume_from_checkpoint=MODEL_NAME if RESUME_FROM_CHECKPOINT else None)
    wandb.finish()
    

if __name__ == "__main__":
    main()