#!/bin/bash
# SafeRec Evaluation Quick Run Script

set -e  # Exit on error

echo "========================================"
echo "SafeRec Model Evaluation"
echo "========================================"
echo ""

# Default values
CHECKPOINT="outputs/checkpoint-100"
NUM_SAMPLES=""
OUTPUT_DIR="eval_results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --quick)
            NUM_SAMPLES="10"
            echo "Quick mode: Evaluating 10 samples"
            shift
            ;;
        --medium)
            NUM_SAMPLES="50"
            echo "Medium mode: Evaluating 50 samples"
            shift
            ;;
        --full)
            NUM_SAMPLES=""
            echo "Full mode: Evaluating all 300 samples"
            shift
            ;;
        --help|-h)
            echo "Usage: ./run_eval.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --checkpoint PATH    Checkpoint directory (default: outputs/checkpoint-100)"
            echo "  --num_samples N      Number of samples to evaluate"
            echo "  --output_dir DIR     Output directory (default: eval_results)"
            echo "  --quick              Evaluate 10 samples (for quick testing)"
            echo "  --medium             Evaluate 50 samples"
            echo "  --full               Evaluate all 300 samples (default)"
            echo "  --help, -h           Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_eval.sh --quick                    # Quick test with 10 samples"
            echo "  ./run_eval.sh --medium                   # Medium test with 50 samples"
            echo "  ./run_eval.sh --full                     # Full evaluation (300 samples)"
            echo "  ./run_eval.sh --checkpoint path/to/ckpt  # Custom checkpoint"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT"
    echo "Please check the path and try again."
    exit 1
fi

# Check for adapter_model.safetensors
if [ ! -f "$CHECKPOINT/adapter_model.safetensors" ]; then
    echo "Warning: adapter_model.safetensors not found in $CHECKPOINT"
    echo "This might not be a valid checkpoint directory."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build command
CMD="/home/coder/miniconda3/envs/safe/bin/python eval.py --checkpoint $CHECKPOINT --output_dir $OUTPUT_DIR"

if [ -n "$NUM_SAMPLES" ]; then
    CMD="$CMD --num_samples $NUM_SAMPLES"
fi

echo ""
echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output Dir: $OUTPUT_DIR"
if [ -n "$NUM_SAMPLES" ]; then
    echo "  Num Samples: $NUM_SAMPLES"
else
    echo "  Num Samples: ALL (300)"
fi
echo ""

# Estimate time
if [ -z "$NUM_SAMPLES" ]; then
    echo "Estimated time: 2.5 - 5 hours"
elif [ "$NUM_SAMPLES" -le 10 ]; then
    echo "Estimated time: 5 - 10 minutes"
elif [ "$NUM_SAMPLES" -le 50 ]; then
    echo "Estimated time: 25 - 50 minutes"
else
    echo "Estimated time: ~$((NUM_SAMPLES * 6 / 60)) - $((NUM_SAMPLES * 12 / 60)) minutes"
fi

echo ""
read -p "Start evaluation? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Evaluation cancelled."
    exit 0
fi

echo ""
echo "Starting evaluation..."
echo "Command: $CMD"
echo ""

# Run evaluation
$CMD

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Evaluation completed successfully!"
    echo "========================================"
    echo ""
    echo "Results saved to: $OUTPUT_DIR/"
    echo ""
    echo "View results:"
    echo "  Summary report:    cat $OUTPUT_DIR/evaluation_summary.md"
    echo "  Statistics:        cat $OUTPUT_DIR/aggregated_statistics.json"
    echo "  Individual sample: cat $OUTPUT_DIR/sample_000.json"
    echo ""
else
    echo ""
    echo "========================================"
    echo "Evaluation failed!"
    echo "========================================"
    echo "Please check the error messages above."
    exit 1
fi
