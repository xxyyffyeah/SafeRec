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
EVAL_MODE="both"  # Options: both, base, checkpoint

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
        --base-only)
            EVAL_MODE="base"
            shift
            ;;
        --checkpoint-only)
            EVAL_MODE="checkpoint"
            shift
            ;;
        --both)
            EVAL_MODE="both"
            shift
            ;;
        --help|-h)
            echo "Usage: ./run_eval.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --checkpoint PATH      Checkpoint directory (default: outputs/checkpoint-100)"
            echo "  --base_model NAME      Base model name (default: unsloth/Qwen3-4B-Instruct-2507)"
            echo "  --num_samples N        Number of samples to evaluate"
            echo "  --output_dir DIR       Output directory (default: eval_results)"
            echo "  --quick                Evaluate 10 samples (for quick testing)"
            echo "  --medium               Evaluate 50 samples"
            echo "  --full                 Evaluate all 300 samples (default)"
            echo "  --base-only            Evaluate only base model"
            echo "  --checkpoint-only      Evaluate only checkpoint model"
            echo "  --both                 Evaluate both models (default)"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_eval.sh --quick                                  # Quick test both models"
            echo "  ./run_eval.sh --checkpoint-only --medium               # Evaluate checkpoint only, 50 samples"
            echo "  ./run_eval.sh --base-only --full                       # Evaluate base only, all samples"
            echo "  ./run_eval.sh --checkpoint outputs/checkpoint-200      # Evaluate both with custom checkpoint"
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
echo "  Eval Mode:   $EVAL_MODE"
if [ "$EVAL_MODE" == "base" ] || [ "$EVAL_MODE" == "both" ]; then
    echo "  Base Model:  $BASE_MODEL"
fi
if [ "$EVAL_MODE" == "checkpoint" ] || [ "$EVAL_MODE" == "both" ]; then
    echo "  Checkpoint:  $CHECKPOINT"
fi
echo "  Output Dir:  $OUTPUT_DIR"
if [ -n "$NUM_SAMPLES" ]; then
    echo "  Num Samples: $NUM_SAMPLES"
else
    echo "  Num Samples: ALL (300)"
fi
echo ""

# Calculate number of models to evaluate
NUM_MODELS=0
if [ "$EVAL_MODE" == "base" ] || [ "$EVAL_MODE" == "both" ]; then
    NUM_MODELS=$((NUM_MODELS + 1))
fi
if [ "$EVAL_MODE" == "checkpoint" ] || [ "$EVAL_MODE" == "both" ]; then
    NUM_MODELS=$((NUM_MODELS + 1))
fi

# Estimate time
if [ -z "$NUM_SAMPLES" ]; then
    echo "Estimated time per model: 2.5 - 5 hours"
    TOTAL_MIN=$((NUM_MODELS * 150))
    TOTAL_MAX=$((NUM_MODELS * 300))
    echo "Total estimated time: $((TOTAL_MIN / 60)) - $((TOTAL_MAX / 60)) hours"
elif [ "$NUM_SAMPLES" -le 10 ]; then
    echo "Estimated time per model: 5 - 10 minutes"
    echo "Total estimated time: $((NUM_MODELS * 5)) - $((NUM_MODELS * 10)) minutes"
elif [ "$NUM_SAMPLES" -le 50 ]; then
    echo "Estimated time per model: 25 - 50 minutes"
    echo "Total estimated time: $((NUM_MODELS * 25)) - $((NUM_MODELS * 50)) minutes"
else
    MINS_PER=$((NUM_SAMPLES * 6 / 60))
    MINS_TOTAL=$((MINS_PER * NUM_MODELS))
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
if [ "$EVAL_MODE" == "base" ] || [ "$EVAL_MODE" == "both" ]; then
    run_eval "$BASE_MODEL" "eval_base.log" "Base Model"
fi

# Evaluate Checkpoint
if [ "$EVAL_MODE" == "checkpoint" ] || [ "$EVAL_MODE" == "both" ]; then
    if [ -d "$CHECKPOINT" ]; then
        run_eval "$CHECKPOINT" "eval_checkpoint.log" "Checkpoint Model"
    else
        echo ""
        echo "Warning: Checkpoint directory not found: $CHECKPOINT"
        echo "Skipping checkpoint evaluation."
    fi
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
