#!/usr/bin/env python3
"""
SafeRec Model Evaluation Script

Usage:
    # Evaluate a checkpoint
    python eval.py --model outputs/checkpoint-100
    python eval.py --model outputs/checkpoint-100 --num_samples 50

    # Evaluate a base model
    python eval.py --model unsloth/Qwen3-4B-Instruct-2507
"""

import argparse
import os
import json
import re
from datetime import datetime
from pathlib import Path
import torch
from tqdm import tqdm

# Import utilities
from utils.rewards import *
from utils.data_loader import load_eval_dataset
from unsloth import FastLanguageModel
from vllm import SamplingParams

# ==================== Configuration ====================
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SafeRec model")
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/checkpoint-100",
        help="Model to evaluate: either a checkpoint path (e.g., outputs/checkpoint-100) or base model name (e.g., unsloth/Qwen3-4B-Instruct-2507)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all 300)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation"
    )
    return parser.parse_args()


# ==================== Model Loading ====================
def load_model(model_path):
    """
    Load model from checkpoint or base model.

    Args:
        model_path: Either a checkpoint path (e.g., outputs/checkpoint-100)
                   or a base model name (e.g., unsloth/Qwen3-4B-Instruct-2507)

    Returns:
        model, tokenizer: Loaded model and tokenizer
    """
    print(f"\nLoading model from {model_path}...")

    max_seq_length = 2048
    lora_rank = 256

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=False,
        max_lora_rank=lora_rank,
    )

    # Enable inference mode
    FastLanguageModel.for_inference(model)

    print("Model loaded successfully!")
    return model, tokenizer


# ==================== Evaluation ====================
def evaluate_sample(model, tokenizer, sample):
    """Evaluate a single sample"""
    # Prepare prompt
    prompt = sample["prompt"]

    # Tokenize and generate
    inputs = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=1500,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode response
    generated_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

    # Extract IMDB IDs from response
    predicted_ids = extract_imdb_ids_from_response(generated_text)

    # Calculate individual rewards
    # Wrap in lists for batch processing compatibility
    completion = [[{"content": generated_text}]]
    prompts = [prompt]
    avoid_content = [sample["avoid_content"]]
    ground_truth_imdb_ids = [sample["ground_truth_imdb_ids"]]

    safety_score = calculate_safety_reward(prompts, completion, avoid_content)[0]
    accuracy_score = calculate_accuracy_reward(prompts, completion, ground_truth_imdb_ids)[0]
    coverage_score = calculate_coverage_reward(prompts, completion)[0]
    final_score = calculate_final_reward(prompts, completion, avoid_content, ground_truth_imdb_ids)[0]

    # Build result dictionary
    result = {
        "prompt": {
            "system": prompt[0]["content"],
            "user": prompt[1]["content"],
        },
        "ground_truth": {
            "answer": sample["answer"],
            "imdb_ids": sample["ground_truth_imdb_ids"],
            "avoid_content": sample["avoid_content"],
        },
        "prediction": {
            "full_response": generated_text,
            "extracted_imdb_ids": predicted_ids,
        },
        "scores": {
            "safety": float(safety_score),
            "accuracy": float(accuracy_score),
            "coverage": float(coverage_score),
            "final": float(final_score),
        },
        "analysis": {
            "num_predicted_movies": len(predicted_ids),
            "num_ground_truth_movies": len(sample["ground_truth_imdb_ids"]),
            "num_correct_predictions": len(set(predicted_ids) & set(sample["ground_truth_imdb_ids"])),
            "response_length": len(generated_text),
        }
    }

    return result


def run_evaluation(model, tokenizer, eval_dataset, output_dir):
    """Run evaluation on entire dataset"""
    print(f"\n{'='*80}")
    print("Starting Evaluation")
    print(f"{'='*80}\n")

    results = []

    # Evaluate each sample
    for idx, sample in enumerate(tqdm(eval_dataset, desc="Evaluating")):
        try:
            result = evaluate_sample(model, tokenizer, sample)
            result["sample_id"] = idx
            results.append(result)

            # Save individual result
            output_file = os.path.join(output_dir, f"sample_{idx:03d}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"\nError evaluating sample {idx}: {e}")
            continue

    return results


# ==================== Results Analysis ====================
def analyze_results(results):
    """Calculate aggregate statistics"""
    if not results:
        return {}

    total = len(results)

    # Aggregate scores
    avg_safety = sum(r["scores"]["safety"] for r in results) / total
    avg_accuracy = sum(r["scores"]["accuracy"] for r in results) / total
    avg_coverage = sum(r["scores"]["coverage"] for r in results) / total
    avg_final = sum(r["scores"]["final"] for r in results) / total

    # Additional statistics
    avg_predicted = sum(r["analysis"]["num_predicted_movies"] for r in results) / total
    avg_ground_truth = sum(r["analysis"]["num_ground_truth_movies"] for r in results) / total
    total_correct = sum(r["analysis"]["num_correct_predictions"] for r in results)
    avg_response_len = sum(r["analysis"]["response_length"] for r in results) / total

    # Score distributions
    safety_scores = [r["scores"]["safety"] for r in results]
    accuracy_scores = [r["scores"]["accuracy"] for r in results]
    coverage_scores = [r["scores"]["coverage"] for r in results]
    final_scores = [r["scores"]["final"] for r in results]

    import statistics

    stats = {
        "total_samples": total,
        "average_scores": {
            "safety": avg_safety,
            "accuracy": avg_accuracy,
            "coverage": avg_coverage,
            "final": avg_final,
        },
        "score_std": {
            "safety": statistics.stdev(safety_scores) if len(safety_scores) > 1 else 0,
            "accuracy": statistics.stdev(accuracy_scores) if len(accuracy_scores) > 1 else 0,
            "coverage": statistics.stdev(coverage_scores) if len(coverage_scores) > 1 else 0,
            "final": statistics.stdev(final_scores) if len(final_scores) > 1 else 0,
        },
        "prediction_stats": {
            "avg_predicted_movies": avg_predicted,
            "avg_ground_truth_movies": avg_ground_truth,
            "total_correct_predictions": total_correct,
            "avg_response_length": avg_response_len,
        },
        "score_distribution": {
            "safety": {
                "min": min(safety_scores),
                "max": max(safety_scores),
                "median": statistics.median(safety_scores),
            },
            "accuracy": {
                "min": min(accuracy_scores),
                "max": max(accuracy_scores),
                "median": statistics.median(accuracy_scores),
            },
            "coverage": {
                "min": min(coverage_scores),
                "max": max(coverage_scores),
                "median": statistics.median(coverage_scores),
            },
            "final": {
                "min": min(final_scores),
                "max": max(final_scores),
                "median": statistics.median(final_scores),
            },
        }
    }

    return stats


def generate_markdown_report(results, stats, output_dir, model_path):
    """Generate a markdown report with summary and examples"""

    report_lines = []
    report_lines.append("# SafeRec Model Evaluation Report")
    report_lines.append(f"\n**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Model**: `{model_path}`")
    report_lines.append(f"**Total Samples Evaluated**: {stats['total_samples']}")

    # Overall Statistics
    report_lines.append("\n## Overall Performance Metrics\n")
    report_lines.append("| Metric | Mean | Std Dev | Min | Max | Median |")
    report_lines.append("|--------|------|---------|-----|-----|--------|")

    for metric in ["safety", "accuracy", "coverage", "final"]:
        mean = stats["average_scores"][metric]
        std = stats["score_std"][metric]
        min_val = stats["score_distribution"][metric]["min"]
        max_val = stats["score_distribution"][metric]["max"]
        median = stats["score_distribution"][metric]["median"]
        report_lines.append(
            f"| {metric.capitalize()} | {mean:.4f} | {std:.4f} | "
            f"{min_val:.4f} | {max_val:.4f} | {median:.4f} |"
        )

    # Prediction Statistics
    report_lines.append("\n## Prediction Statistics\n")
    pred_stats = stats["prediction_stats"]
    report_lines.append(f"- **Average Predicted Movies**: {pred_stats['avg_predicted_movies']:.2f}")
    report_lines.append(f"- **Average Ground Truth Movies**: {pred_stats['avg_ground_truth_movies']:.2f}")
    report_lines.append(f"- **Total Correct Predictions**: {pred_stats['total_correct_predictions']}")
    report_lines.append(f"- **Average Response Length**: {pred_stats['avg_response_length']:.0f} characters")

    # Score Breakdown Analysis
    report_lines.append("\n## Score Breakdown Analysis\n")
    report_lines.append("### Safety Score")
    report_lines.append(f"- Measures whether recommended movies avoid user's sensitivity triggers")
    report_lines.append(f"- Formula: 0.6 × (Top-5 safety) + 0.4 × (Positions 6-20 safety)")
    report_lines.append(f"- **Result**: {stats['average_scores']['safety']:.2%} average safety rate")

    report_lines.append("\n### Accuracy Score")
    report_lines.append(f"- Measures overlap with ground truth recommendations")
    report_lines.append(f"- Formula: Number of GT hits / Total GT count")
    report_lines.append(f"- **Result**: {stats['average_scores']['accuracy']:.2%} average accuracy")

    report_lines.append("\n### Coverage Score")
    report_lines.append(f"- Measures proportion of valid IMDB IDs in recommendations")
    report_lines.append(f"- Formula: Valid IMDB IDs / Total recommendations")
    report_lines.append(f"- **Result**: {stats['average_scores']['coverage']:.2%} average coverage")

    report_lines.append("\n### Final Score")
    report_lines.append(f"- Combined weighted score: 0.7 × Safety + 0.2 × Accuracy + 0.1 × Coverage")
    report_lines.append(f"- Scaled by 10x for training signal")
    report_lines.append(f"- **Result**: {stats['average_scores']['final']:.4f} / 10.0")

    # Select 5 representative examples
    report_lines.append("\n## Sample Predictions\n")
    report_lines.append("Below are 5 representative examples from the evaluation:\n")

    # Sort by final score to get diverse examples
    sorted_results = sorted(results, key=lambda x: x["scores"]["final"])

    # Select: worst, 25th percentile, median, 75th percentile, best
    indices = [
        0,  # worst
        len(sorted_results) // 4,  # 25th percentile
        len(sorted_results) // 2,  # median
        3 * len(sorted_results) // 4,  # 75th percentile
        -1,  # best
    ]

    for i, idx in enumerate(indices, 1):
        example = sorted_results[idx]
        percentile = ["Worst", "25th Percentile", "Median", "75th Percentile", "Best"][i-1]

        report_lines.append(f"### Example {i}: {percentile} Performance\n")
        report_lines.append(f"**Sample ID**: {example['sample_id']}")
        report_lines.append(f"**Scores**: Safety={example['scores']['safety']:.4f}, "
                          f"Accuracy={example['scores']['accuracy']:.4f}, "
                          f"Coverage={example['scores']['coverage']:.4f}, "
                          f"Final={example['scores']['final']:.4f}\n")

        # User query
        user_query = example["prompt"]["user"]
        report_lines.append(f"**User Query**:")
        report_lines.append(f"```")
        report_lines.append(user_query[:500] + ("..." if len(user_query) > 500 else ""))
        report_lines.append(f"```\n")

        # Avoid content
        report_lines.append(f"**User's Content Sensitivities**:")
        for content in example["ground_truth"]["avoid_content"][:5]:
            report_lines.append(f"- {content}")
        if len(example["ground_truth"]["avoid_content"]) > 5:
            report_lines.append(f"- ... and {len(example['ground_truth']['avoid_content']) - 5} more")
        report_lines.append("")

        # Ground truth
        report_lines.append(f"**Ground Truth IMDB IDs**: {example['ground_truth']['imdb_ids']}\n")

        # Model prediction
        report_lines.append(f"**Model Response**:")
        report_lines.append(f"```")
        response = example["prediction"]["full_response"]
        report_lines.append(response)
        report_lines.append(f"```\n")

        # Predicted IDs
        report_lines.append(f"**Extracted IMDB IDs**: {example['prediction']['extracted_imdb_ids'][:10]}")
        if len(example['prediction']['extracted_imdb_ids']) > 10:
            report_lines.append(f"... and {len(example['prediction']['extracted_imdb_ids']) - 10} more")
        report_lines.append("")

        # Analysis
        report_lines.append(f"**Analysis**:")
        report_lines.append(f"- Predicted {example['analysis']['num_predicted_movies']} movies")
        report_lines.append(f"- {example['analysis']['num_correct_predictions']} correct matches with ground truth")
        report_lines.append(f"- Response length: {example['analysis']['response_length']} characters")
        report_lines.append("\n" + "-" * 80 + "\n")

    # Save report
    report_path = os.path.join(output_dir, "evaluation_summary.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"\nMarkdown report saved to: {report_path}")

    return report_path


# ==================== Main ====================
def main():
    args = parse_args()

    # Create output directory with timestamp and model name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Use model path/name directly, replacing slashes for directory naming
    model_name = args.model.replace('/', '_').replace('\\', '_')

    # Create unique run directory
    base_output_dir = args.output_dir
    run_dir_name = f"{timestamp}_{model_name}"
    output_dir = os.path.join(base_output_dir, run_dir_name)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load dataset
    eval_dataset = load_eval_dataset(args.num_samples)

    # Load model
    model, tokenizer = load_model(args.model)

    # Run evaluation
    results = run_evaluation(model, tokenizer, eval_dataset, output_dir)

    # Check if evaluation produced any results
    if not results:
        print("\n" + "="*80)
        print("ERROR: No evaluation results generated!")
        return

    # Analyze results
    print("\nAnalyzing results...")
    stats = analyze_results(results)

    # Save aggregated statistics
    stats_file = os.path.join(output_dir, "aggregated_statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Statistics saved to: {stats_file}")

    # Generate markdown report
    report_path = generate_markdown_report(results, stats, output_dir, args.model)

    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nTotal Samples: {stats['total_samples']}")
    print(f"\nAverage Scores:")
    print(f"  Safety:   {stats['average_scores']['safety']:.4f}")
    print(f"  Accuracy: {stats['average_scores']['accuracy']:.4f}")
    print(f"  Coverage: {stats['average_scores']['coverage']:.4f}")
    print(f"  Final:    {stats['average_scores']['final']:.4f}")
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - Individual results: sample_*.json")
    print(f"  - Aggregated stats: aggregated_statistics.json")
    print(f"  - Summary report: evaluation_summary.md")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
