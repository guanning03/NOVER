import os
from typing import Optional, Tuple, Dict, Any
from datasets import Dataset, load_from_disk, load_dataset
from config import DATASET_NAME, DATASET_SUBSET, get_dataset_path

def load_dataset(
    dataset_name: str = DATASET_NAME,
    subset: Optional[int] = DATASET_SUBSET,
    shuffle: bool = True,
    seed: int = 42,
    val_size: int = 0
) -> Dataset:
    dataset_path = get_dataset_path(dataset_name)
    dataset = load_from_disk(dataset_path)
    
    train_data = dataset["train"]
    
    if shuffle:
        train_data = train_data.shuffle(seed=seed)
    
    if subset is not None and isinstance(subset, int) and subset > 0:
        train_data = train_data.select(range(min(subset, len(train_data))))
    
    return train_data

def load_train_val_dataset(
    dataset_name: str = DATASET_NAME,
    subset: Optional[int] = DATASET_SUBSET,
    shuffle: bool = True,
    seed: int = 42,
    val_size: int = 128
) -> Tuple[Dataset, Dataset]:
    dataset_path = get_dataset_path(dataset_name)
    dataset = load_from_disk(dataset_path)
    
    has_validation_split = "validation" in dataset
    
    full_data = dataset["train"]
    
    if shuffle:
        full_data = full_data.shuffle(seed=seed)
    
    if subset is not None and isinstance(subset, int) and subset > 0:
        full_data = full_data.select(range(min(subset, len(full_data))))
    
    if val_size <= 0:
        print(f"no validation")
        return full_data, None
    
    if has_validation_split:
        val_data = dataset["test"]
        
        val_subset_size = min(val_size, len(val_data))
        val_data = val_data.select(range(val_subset_size))
        
        if shuffle:
            val_data = val_data.shuffle(seed=seed)
            
        print(f"[INFO] Using existing validation split: {len(val_data)} examples (limited to {val_size})")
        return full_data, val_data
    
    train_size = len(full_data) - val_size
    train_data = full_data.select(range(train_size))
    val_data = full_data.select(range(train_size, len(full_data)))
    
    print(f"[INFO] Created validation split from training data: {len(train_data)} train, {len(val_data)} validation")
    return train_data, val_data

def format_sft_dataset(
    dataset_source: str,
    prompt_column: str = "prompt",
    reference_column: str = "reference",
    output_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    split: str = "train"
) -> str:
    """
    Format any SFT dataset (HF/CSV/JSONL) for NOVER training.
    
    Args:
        dataset_source: HF dataset name or local file path
        prompt_column: Column containing questions/prompts
        reference_column: Column containing reference answers
        output_dir: Directory to save formatted dataset
        cache_dir: Cache directory for HF datasets
        split: Dataset split to use
        
    Returns:
        Path to the formatted dataset
    """
    # Determine if it's a local file or HF dataset
    is_local_file = os.path.exists(dataset_source)
    
    # Load the dataset
    if is_local_file:
        if dataset_source.endswith('.csv'):
            dataset = load_dataset('csv', data_files=dataset_source, split=split)
        elif dataset_source.endswith('.jsonl') or dataset_source.endswith('.json'):
            dataset = load_dataset('json', data_files=dataset_source, split=split)
        else:
            raise ValueError(f"Unsupported file format: {dataset_source}")
    else:
        dataset = load_dataset(dataset_source, split=split, cache_dir=cache_dir)
    
    # Check columns exist
    if prompt_column not in dataset.column_names:
        raise ValueError(f"Prompt column '{prompt_column}' not found in dataset. Available columns: {dataset.column_names}")
    
    if reference_column not in dataset.column_names:
        raise ValueError(f"Reference column '{reference_column}' not found in dataset. Available columns: {dataset.column_names}")
    
    # Format dataset with NOVER prompt template
    def format_example(example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = example[prompt_column]
        reference = example[reference_column]
        
        # Apply NOVER prompt template
        formatted_prompt = f"""Question: {prompt}

Answer the question and return in the following format:

<think>
...
</think>

<answer>
...
</answer>
"""
        return {
            "prompt": formatted_prompt,
            "reference": reference
        }
    
    formatted_dataset = dataset.map(format_example)
    
    # Keep only needed columns
    formatted_dataset = formatted_dataset.remove_columns(
        [col for col in formatted_dataset.column_names if col not in ["prompt", "reference"]]
    )
    
    # Save formatted dataset
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        formatted_dataset = {"train": formatted_dataset}
        output_path = os.path.join(output_dir, "formatted_dataset")
        Dataset.from_dict(formatted_dataset["train"]).save_to_disk(output_path)
        print(f"[INFO] Formatted dataset saved to {output_path}")
        return output_path
    
    return formatted_dataset