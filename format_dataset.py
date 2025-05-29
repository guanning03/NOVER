#!/usr/bin/env python3
import argparse
import os
import sys
from data_loader import format_sft_dataset
from config import HF_HOME, get_dataset_path, INTERMEDIATE_TAG, FINAL_TAG

def main():
    parser = argparse.ArgumentParser(description="Format any SFT dataset for NOVER training")
    
    parser.add_argument(
        "dataset_source",
        type=str,
        help="Path to dataset file (CSV/JSONL/JSON) or HuggingFace dataset name"
    )
    
    parser.add_argument(
        "--prompt-column",
        type=str,
        default="prompt",
        help="Column name containing the prompts/questions"
    )
    
    parser.add_argument(
        "--reference-column",
        type=str,
        default="reference",
        help="Column name containing reference answers"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: HF_HOME/datasets/dataset_name)"
    )
    
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name for saving (default: derived from source)"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use"
    )
    
    parser.add_argument(
        "--intermediate-tag",
        type=str,
        default=INTERMEDIATE_TAG,
        help=f"Custom intermediate tag (default: {INTERMEDIATE_TAG})"
    )
    
    parser.add_argument(
        "--final-tag",
        type=str,
        default=FINAL_TAG,
        help=f"Custom final tag (default: {FINAL_TAG})"
    )
    
    args = parser.parse_args()
    
    # Derive dataset name from source if not provided
    if args.dataset_name is None:
        if os.path.exists(args.dataset_source):
            args.dataset_name = os.path.splitext(os.path.basename(args.dataset_source))[0]
        else:
            args.dataset_name = args.dataset_source.replace("/", "_")
    
    # Set output directory if not provided
    if args.output_dir is None:
        args.output_dir = os.path.join(HF_HOME, "datasets", args.dataset_name)
    
    print(f"Formatting dataset from: {args.dataset_source}")
    print(f"Using prompt column: {args.prompt_column}")
    print(f"Using reference column: {args.reference_column}")
    print(f"Using intermediate tag: <{args.intermediate_tag}>")
    print(f"Using final tag: <{args.final_tag}>")
    
    try:
        # Format and save the dataset
        dataset_path = format_sft_dataset(
            dataset_source=args.dataset_source,
            prompt_column=args.prompt_column,
            reference_column=args.reference_column,
            output_dir=args.output_dir,
            split=args.split,
            intermediate_tag=args.intermediate_tag,
            final_tag=args.final_tag
        )
        
        print(f"Dataset successfully formatted and saved to: {dataset_path}")
        print("\nTo use this dataset for NOVER training, update your config.py:")
        print(f'DATASET_NAME = "{args.dataset_name}"')
        print(f'INTERMEDIATE_TAG = "{args.intermediate_tag}"')
        print(f'FINAL_TAG = "{args.final_tag}"')
        
    except Exception as e:
        print(f"Error formatting dataset: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 