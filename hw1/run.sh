#!/bin/bash

# Combined inference script for ADL HW1
# First runs multiple choice to select best paragraphs, then QA to extract answers

# Check if correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <context.json> <test.json> <prediction.csv>"
    echo "Example: bash run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv"
    exit 1
fi

echo "Starting Combined Inference Pipeline..."
echo "=================================="

# Get paths from command line arguments
CONTEXT_FILE="$1"
TEST_FILE="$2"
OUTPUT_FILE="$3"

# Fixed model paths
MC_MODEL_PATH="./ckpt/mq/chinese_lert"  # Multiple Choice model checkpoint
QA_MODEL_PATH="./ckpt/qa/chinese_lert"  # Question Answering model checkpoint

# Check if input files exist
if [ ! -f "$CONTEXT_FILE" ]; then
    echo "Error: Context file '$CONTEXT_FILE' not found!"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "Error: Test file '$TEST_FILE' not found!"
    exit 1
fi

# Check if model checkpoints exist
if [ ! -d "$MC_MODEL_PATH" ]; then
    echo "Error: Multiple Choice model checkpoint '$MC_MODEL_PATH' not found!"
    exit 1
fi

if [ ! -d "$QA_MODEL_PATH" ]; then
    echo "Error: Question Answering model checkpoint '$QA_MODEL_PATH' not found!"
    exit 1
fi

echo "Input files:"
echo "  Context file: $CONTEXT_FILE"
echo "  Test file: $TEST_FILE"
echo "  Output file: $OUTPUT_FILE"
echo "Model checkpoints:"
echo "  MC model: $MC_MODEL_PATH"
echo "  QA model: $QA_MODEL_PATH"
echo "----------------------------------"

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"

# Run combined inference
CUDA_VISIBLE_DEVICES=2 python inference.py \
    --test_file "$TEST_FILE" \
    --context_file "$CONTEXT_FILE" \
    --mc_model_path "$MC_MODEL_PATH" \
    --qa_model_path "$QA_MODEL_PATH" \
    --output_file "$OUTPUT_FILE" \
    --batch_size 8 \
    --max_length 512 \
    --device cuda

echo "=================================="
echo "Combined inference completed!"
echo "Final results saved to: $OUTPUT_FILE"
echo "Intermediate MC predictions saved to: $OUTPUT_DIR/mc_predictions.csv"