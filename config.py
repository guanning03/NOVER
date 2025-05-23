import os
import torch
import wandb
import yaml
import glob
from datetime import datetime
from trl import GRPOConfig
from peft import LoraConfig

#######################
# Basic Configuration #
#######################

# Project identification
SUFFIX = "YOUR_SUFFIX"
WANDB_PROJECT = "YOUR_WANDB_PROJECT"

# Dataset settings
DATASET_NAME = "YOUR_DATASET_NAME_UNDER_HF_HOME"
DATASET_SUBSET = None
HF_HOME = os.environ.get("HF_HOME", "YOUR_HF_HOME")
VALIDATION_SIZE = 0

# Model settings
MODEL_NAME_VLLM = "YOUR_MODEL_NAME_VLLM"
MODEL_NAME = "YOUR_MODEL_NAME"
SAVE_BASE_PATH = "YOUR_SAVE_BASE_PATH"

# Checkpoint settings
RESUME_FROM_CHECKPOINT = False
ADAPTER_DIR = ""
LATEST_CHECKPOINT_STEP = None

#########################
# Training Hyperparams  #
#########################

# Batch settings
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1
NUM_GENERATIONS = 8

# Training loop settings
NUM_TRAIN_EPOCHS = 2
NUM_ITERATIONS = 1
LEARNING_RATE = 1e-5
BETA = 0.1

# Checkpoint and logging
LOGGING_STEPS = 20
SAVE_STEPS = 100
EVAL_STEPS = 100
SAVE_TOTAL_LIMIT = 30
EVALUATION_STRATEGY = "no"

# Generation settings
MAX_COMPLETION_LENGTH = 2048
TEMPERATURE = 0.6

# GRPO specific settings
SCALE_REWARDS = False
EPSILON = 0.1
EPSILON_HIGH = 0.2

# Reference model settings
SYNC_REF_MODEL = True
REF_MODEL_MIXUP_ALPHA = 0.9

#####################
# Reward Functions  #
#####################

TAG_FORMAT_REWARD_WEIGHT = 1.0
REASONING_REWARD_WEIGHT = 1.0
EFFICIENCY_REWARD_WEIGHT = 1.0
REWARD_WEIGHTS = [TAG_FORMAT_REWARD_WEIGHT, REASONING_REWARD_WEIGHT, EFFICIENCY_REWARD_WEIGHT]

#######################
# VLLM Configuration  #
#######################

USE_VLLM = True
VLLM_HOST = "localhost"
VLLM_PORT = 8087
VLLM_GPU_MEMORY_UTILIZATION = 0.8
VLLM_REQUEST_TIMEOUT = 3600

#######################
# LoRA Configuration  #
#######################

PEFT_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    modules_to_save=["lm_head"],
)

#######################
# Helper Functions    #
#######################

def find_latest_checkpoint():
    """
    Find the latest checkpoint in the adapter directory if resuming training.
    Returns the step number of the latest checkpoint or None if not found.
    """
    if not RESUME_FROM_CHECKPOINT or not ADAPTER_DIR or not os.path.exists(ADAPTER_DIR):
        return None
    
    checkpoint_dirs = sorted(glob.glob(os.path.join(ADAPTER_DIR, "checkpoint-*")))
    if not checkpoint_dirs:
        return None
    
    steps = []
    for dir_path in checkpoint_dirs:
        try:
            step = int(dir_path.split("-")[-1])
            steps.append(step)
        except ValueError:
            continue
    
    if not steps:
        return None
    
    latest_step = max(steps)
    print(f"Found latest checkpoint at step {latest_step}")
    return latest_step

# Initialize checkpoint step if resuming training
if RESUME_FROM_CHECKPOINT:
    LATEST_CHECKPOINT_STEP = find_latest_checkpoint()

# Update model path if resuming from checkpoint
if RESUME_FROM_CHECKPOINT and LATEST_CHECKPOINT_STEP:
    MODEL_NAME = os.path.join(ADAPTER_DIR, f"checkpoint-{LATEST_CHECKPOINT_STEP}")
    print(f"Resuming training from checkpoint: {MODEL_NAME}")

def setup_environment():
    """
    Set up environment variables for distributed training and clear GPU memory.
    """
    os.environ.update({
        "NCCL_DEBUG": "INFO",
        "NCCL_TIMEOUT": "3600",
        "NCCL_SOCKET_TIMEOUT": "3600",
        "NCCL_IB_TIMEOUT": "3600",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "NCCL_P2P_DISABLE": "1",  
        "NCCL_IB_DISABLE": "0",
        "NCCL_SOCKET_IFNAME": "^docker0,lo",
    })
    torch.cuda.empty_cache()

def get_precision_config():
    """
    Determine the appropriate precision settings based on available hardware.
    Returns a tuple of (precision_type, torch_dtype, vllm_dtype).
    """
    if not torch.cuda.is_available():
        return "fp16", torch.float16, "half"
    
    gpu_props = torch.cuda.get_device_properties(0)
    compute_capability = float(f"{gpu_props.major}.{gpu_props.minor}")
    
    return ("bf16", torch.bfloat16, "bfloat16") if compute_capability >= 8.0 else ("fp16", torch.float16, "half")

def get_output_dir(suffix=""):
    """
    Generate the output directory path for saving model checkpoints.
    If resuming training, uses the adapter directory.
    Otherwise, creates a new directory with timestamp.
    """
    if RESUME_FROM_CHECKPOINT and ADAPTER_DIR:
        return ADAPTER_DIR
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    model_name_short = MODEL_NAME.split('/')[-1]
    base_dir = SAVE_BASE_PATH
    dir_name = f"{model_name_short}_{timestamp}"
    if suffix:
        dir_name += f"_{suffix}"
    return os.path.join(base_dir, dir_name)

# Set output directory
OUTPUT_DIR = get_output_dir(suffix=SUFFIX)

def init_wandb():
    """
    Initialize Weights & Biases logging with appropriate configuration.
    Restores original environment variables on exit.
    """
    my_wandb_api_key = None
    my_wandb_entity = None
    
    import os
    original_api_key = os.environ.get("WANDB_API_KEY")
    original_entity = os.environ.get("WANDB_ENTITY")
    
    if my_wandb_api_key:
        os.environ["WANDB_API_KEY"] = my_wandb_api_key
    if my_wandb_entity:
        os.environ["WANDB_ENTITY"] = my_wandb_entity
    
    model_short = MODEL_NAME.split('/')[-1]
    timestamp = datetime.now().strftime('%m%d_%H%M')
    
    if RESUME_FROM_CHECKPOINT:
        run_name = f"{model_short}_r{LATEST_CHECKPOINT_STEP}_{timestamp}"
    else:
        run_name = f"{model_short}_{timestamp}"
    
    if SUFFIX:
        run_name = f"{SUFFIX}_{run_name}"
    
    wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config={
            "model": MODEL_NAME,
            "output_dir": OUTPUT_DIR,
            "resume_from_checkpoint": RESUME_FROM_CHECKPOINT,
            "latest_checkpoint_step": LATEST_CHECKPOINT_STEP if RESUME_FROM_CHECKPOINT else None,
            "adapter_dir": ADAPTER_DIR if RESUME_FROM_CHECKPOINT else None,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "epochs": NUM_TRAIN_EPOCHS,
            "num_iterations": NUM_ITERATIONS,
            "num_generations": NUM_GENERATIONS,
            "beta": BETA,
            "logging_steps": LOGGING_STEPS,
            "save_steps": SAVE_STEPS,
            "save_total_limit": SAVE_TOTAL_LIMIT,
            "scale_rewards": SCALE_REWARDS,
            "epsilon": EPSILON,
            "epsilon_high": EPSILON_HIGH,
            "learning_rate": LEARNING_RATE,
            "temperature": TEMPERATURE,
            "max_completion_length": MAX_COMPLETION_LENGTH,
            "sync_ref_model": SYNC_REF_MODEL,
            "ref_model_mixup_alpha": REF_MODEL_MIXUP_ALPHA,
            "use_vllm": USE_VLLM,
            "vllm_host": VLLM_HOST if USE_VLLM else None,
            "vllm_port": VLLM_PORT if USE_VLLM else None,
            "vllm_gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
            "vllm_request_timeout": VLLM_REQUEST_TIMEOUT,
            "reward_weights": {
                "tag_format_reward": TAG_FORMAT_REWARD_WEIGHT,
                "reasoning_reward": REASONING_REWARD_WEIGHT,
                "efficiency_reward": EFFICIENCY_REWARD_WEIGHT
            },
            "peft_config": {
                "r": PEFT_CONFIG.r,
                "lora_alpha": PEFT_CONFIG.lora_alpha,
                "lora_dropout": PEFT_CONFIG.lora_dropout,
                "bias": PEFT_CONFIG.bias,
                "task_type": PEFT_CONFIG.task_type
            },
            "dataset": {
                "name": DATASET_NAME,
                "subset": DATASET_SUBSET,
                "validation_size": VALIDATION_SIZE
            }
        }
    )
    
    def restore_env_vars():
        """Restore original environment variables when exiting."""
        if my_wandb_api_key and original_api_key is not None:
            os.environ["WANDB_API_KEY"] = original_api_key
        elif my_wandb_api_key and original_api_key is None:
            os.environ.pop("WANDB_API_KEY", None)
            
        if my_wandb_entity and original_entity is not None:
            os.environ["WANDB_ENTITY"] = original_entity
        elif my_wandb_entity and original_entity is None:
            os.environ.pop("WANDB_ENTITY", None)
    
    import atexit
    atexit.register(restore_env_vars)

def get_training_config():
    """
    Create and return the GRPO training configuration object.
    """
    precision_type, torch_dtype, vllm_dtype = get_precision_config()
    
    config = GRPOConfig(
        # Output settings
        output_dir=OUTPUT_DIR,
        
        # Batch settings
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        dataloader_drop_last=True,
        
        # Optimization settings
        learning_rate=LEARNING_RATE,
        gradient_checkpointing=True,
        **{precision_type: True},
        
        # Training loop settings
        num_train_epochs=NUM_TRAIN_EPOCHS,
        num_iterations=NUM_ITERATIONS,
        beta=BETA,
        
        # Checkpoint and logging
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        evaluation_strategy=EVALUATION_STRATEGY,
        eval_steps=EVAL_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to=["wandb"],
        log_completions=True,
        
        # Model initialization
        model_init_kwargs={
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True
        },
        
        # GRPO specific settings
        num_generations=NUM_GENERATIONS,
        temperature=TEMPERATURE,
        epsilon=EPSILON,
        epsilon_high=EPSILON_HIGH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        reward_weights=REWARD_WEIGHTS,
        
        # Reference model settings
        sync_ref_model=SYNC_REF_MODEL,
        ref_model_mixup_alpha=REF_MODEL_MIXUP_ALPHA,
        
        # VLLM settings
        use_vllm=USE_VLLM,
        vllm_server_host=VLLM_HOST if USE_VLLM else None,
        vllm_server_port=VLLM_PORT if USE_VLLM else None,
        vllm_server_timeout=VLLM_REQUEST_TIMEOUT if USE_VLLM else None,
    )
    
    return config

def get_dataset_path(dataset_name=DATASET_NAME):
    """Return the full path to the dataset."""
    return os.path.join(HF_HOME, "datasets", dataset_name)

def save_config_to_yaml():
    """
    Save the complete configuration to a YAML file in the output directory.
    """
    precision_type, torch_dtype, vllm_dtype = get_precision_config()
    
    config = {
        "model": {
            "name": MODEL_NAME,
            "vllm_name": MODEL_NAME_VLLM,
            "output_dir": OUTPUT_DIR,
            "resume_from_checkpoint": RESUME_FROM_CHECKPOINT,
            "latest_checkpoint_step": LATEST_CHECKPOINT_STEP if RESUME_FROM_CHECKPOINT else None,
            "adapter_dir": ADAPTER_DIR if RESUME_FROM_CHECKPOINT else None,
            "suffix": SUFFIX,
            "save_base_path": SAVE_BASE_PATH
        },
        "dataset": {
            "name": DATASET_NAME,
            "subset": DATASET_SUBSET,
            "hf_home": HF_HOME,
            "validation_size": VALIDATION_SIZE
        },
        "training": {
            "wandb_project": WANDB_PROJECT,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "num_train_epochs": NUM_TRAIN_EPOCHS,
            "num_iterations": NUM_ITERATIONS,
            "beta": BETA,
            "logging_steps": LOGGING_STEPS,
            "save_steps": SAVE_STEPS,
            "save_total_limit": SAVE_TOTAL_LIMIT,
            "num_generations": NUM_GENERATIONS,
            "scale_rewards": SCALE_REWARDS,
            "epsilon": EPSILON,
            "epsilon_high": EPSILON_HIGH,
            "max_completion_length": MAX_COMPLETION_LENGTH,
            "dataloader_drop_last": True,
            "gradient_checkpointing": True,
            "precision": precision_type,
            "save_strategy": "steps",
            "evaluation_strategy": "steps",
            "eval_steps": EVAL_STEPS,
            "model_init_kwargs": {
                "torch_dtype": str(torch_dtype),
                "low_cpu_mem_usage": True
            },
            "log_completions": True,
            "report_to": ["wandb"],
            "reward_weights": {
                "tag_format_reward": TAG_FORMAT_REWARD_WEIGHT,
                "reasoning_reward": REASONING_REWARD_WEIGHT,
                "efficiency_reward": EFFICIENCY_REWARD_WEIGHT,
                "weights_list": REWARD_WEIGHTS
            }
        },
        "reference_model": {
            "sync_ref_model": SYNC_REF_MODEL,
            "ref_model_mixup_alpha": REF_MODEL_MIXUP_ALPHA,
        },
        "vllm": {
            "enabled": USE_VLLM,
            "host": VLLM_HOST if USE_VLLM else None,
            "port": VLLM_PORT if USE_VLLM else None,
            "temperature": TEMPERATURE,
            "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
            "request_timeout": VLLM_REQUEST_TIMEOUT
        },
        "peft": {
            "r": PEFT_CONFIG.r,
            "lora_alpha": PEFT_CONFIG.lora_alpha,
            "lora_dropout": PEFT_CONFIG.lora_dropout,
            "bias": PEFT_CONFIG.bias,
            "task_type": PEFT_CONFIG.task_type
        },
        "environment": {
            "nccl_debug": "INFO",
            "nccl_timeout": "3600",
            "nccl_socket_timeout": "3600",
            "nccl_ib_timeout": "3600",
            "pytorch_cuda_alloc_conf": "expandable_segments:True",
            "nccl_p2p_disable": "1",
            "nccl_ib_disable": "0",
            "nccl_socket_ifname": "^docker0,lo"
        }
    }
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config_path = os.path.join(OUTPUT_DIR, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
