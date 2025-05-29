#!/bin/bash

# Make the format_dataset.py script executable
chmod +x format_dataset.py

# Check if any arguments were provided
if [ $# -eq 0 ]; then
    echo "Usage: ./format_dataset.sh [DATASET_SOURCE] [OPTIONS]"
    echo ""
    echo "Examples:"
    echo "  # Format Hugging Face dataset"
    echo "  ./format_dataset.sh squad --prompt-column question --reference-column answers.text"
    echo ""
    echo "  # Format local CSV file"
    echo "  ./format_dataset.sh data.csv --prompt-column question --reference-column answer"
    echo ""
    echo "  # Format local JSONL file"
    echo "  ./format_dataset.sh data.jsonl --prompt-column input --reference-column output"
    echo ""
    echo "For more options, run: ./format_dataset.py --help"
    exit 1
fi

# Run the Python script with all arguments passed to this shell script
python3 format_dataset.py "$@" 