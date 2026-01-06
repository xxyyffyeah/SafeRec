#!/bin/bash
# SafeRec Evaluation Script - Evaluate Checkpoint and Base Model

set -e  # Exit on error

echo "========================================"
echo "SafeRec Model Evaluation"
echo "Evaluating Checkpoint and Base Model"
echo "========================================"
echo ""

# Configuration
BASE_MODEL="unsloth/Qwen3-4B-Instruct-2507"
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
        --base_model)
            BASE_MODEL="$2"
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
            echo "  --base_model NAME    Base model name (default: unsloth/Qwen2.5-1.5B)"
            echo "  --num_samples N      Number of samples to evaluate"
            echo "  --output_dir DIR     Output directory (default: eval_results)"
            echo "  --quick              Evaluate 10 samples (for quick testing)"
            echo "  --medium             Evaluate 50 samples"
            echo "  --full               Evaluate all 300 samples (default)"
            echo "  --help, -h           Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_eval.sh --quick                              # Quick test"
            echo "  ./run_eval.sh --medium                             # Medium test"
            echo "  ./run_eval.sh --full                               # Full evaluation"
            echo "  ./run_eval.sh --checkpoint outputs/checkpoint-200  # Custom checkpoint"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build base command options
SAMPLE_OPT=""
if [ -n "$NUM_SAMPLES" ]; then
    SAMPLE_OPT="--num_samples $NUM_SAMPLES"
fi

echo "Configuration:"
echo "  Base Model:  $BASE_MODEL"
echo "  Checkpoint:  $CHECKPOINT"
echo "  Output Dir:  $OUTPUT_DIR"
if [ -n "$NUM_SAMPLES" ]; then
    echo "  Num Samples: $NUM_SAMPLES"
else
    echo "  Num Samples: ALL (300)"
fi
echo ""

# Estimate time
if [ -z "$NUM_SAMPLES" ]; then
    echo "Estimated time per model: 2.5 - 5 hours"
    echo "Total estimated time: 5 - 10 hours"
elif [ "$NUM_SAMPLES" -le 10 ]; then
    echo "Estimated time per model: 5 - 10 minutes"
    echo "Total estimated time: 10 - 20 minutes"
elif [ "$NUM_SAMPLES" -le 50 ]; then
    echo "Estimated time per model: 25 - 50 minutes"
    echo "Total estimated time: 50 - 100 minutes"
else
    MINS_PER=$((NUM_SAMPLES * 6 / 60))
    MINS_TOTAL=$((MINS_PER * 2))
    echo "Estimated time per model: ~$MINS_PER - $((MINS_PER * 2)) minutes"
    echo "Total estimated time: ~$MINS_TOTAL - $((MINS_TOTAL * 2)) minutes"
fi

echo ""
read -p "Start evaluation? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Evaluation cancelled."
    exit 0
fi

# Function to run evaluation
run_eval() {
    local MODEL=$1
    local LOG_FILE=$2
    local DESC=$3

    echo ""
    echo "========================================"
    echo "Evaluating: $DESC"
    echo "Model: $MODEL"
    echo "========================================"
    echo ""

    CMD="/home/coder/miniconda3/envs/safe/bin/python eval.py --model $MODEL --output_dir $OUTPUT_DIR $SAMPLE_OPT"
    echo "Command: $CMD"
    echo "Logging to: $LOG_FILE"
    echo ""

    # Run with nohup and tee for logging
    nohup $CMD 2>&1 | tee $LOG_FILE

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ $DESC evaluation completed successfully!"
    else
        echo ""
        echo "✗ $DESC evaluation failed!"
        echo "Check log: $LOG_FILE"
        return 1
    fi
}

# Evaluate Base Model
run_eval "$BASE_MODEL" "eval_base.log" "Base Model"

# Evaluate Checkpoint
if [ -d "$CHECKPOINT" ]; then
    run_eval "$CHECKPOINT" "eval_checkpoint.log" "Checkpoint Model"
else
    echo ""
    echo "Warning: Checkpoint directory not found: $CHECKPOINT"
    echo "Skipping checkpoint evaluation."
fi

# Summary
echo ""
echo "========================================"
echo "All Evaluations Completed!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Logs:"
echo "  Base model:       eval_base.log"
echo "  Checkpoint model: eval_checkpoint.log"
echo ""
echo "Compare results in: $OUTPUT_DIR/"
