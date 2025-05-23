import os
from typing import Optional, Tuple
from datasets import Dataset, load_from_disk
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