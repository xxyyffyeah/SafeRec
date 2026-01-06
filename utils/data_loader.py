"""
Data loading utilities for SafeRec training and evaluation.

This module provides shared dataset loading functions used by both
reinforce.py (training) and eval.py (evaluation) to ensure consistency.
"""

from datasets import load_dataset
from utils.prompts import system_prompt


def load_and_prepare_dataset(test_size=300, seed=3407):
    """
    Load and split dataset for training and evaluation.

    This function loads the Reddit movie recommendation dataset and applies
    the same transformations used in both training and evaluation scripts.

    Args:
        test_size: Number of samples to reserve for testing (default: 300)
        seed: Random seed for reproducible splitting (default: 3407)

    Returns:
        A dictionary with keys 'train' and 'test', each containing a Dataset object
    """
    print("Loading dataset...")
    dataset = load_dataset("Dionysianspirit/reddit-movie-rec-sft", split="train")

    # Apply transformations to match expected format
    dataset = dataset.map(lambda x: {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"USER TRAIT: {x['assigned_trait']}\n\n{x['context']}"},
        ],
        "answer": x["ground_truth"],  # ground_truth is already a list of movie names
        "avoid_content": x["avoid_content"],
        "ground_truth_imdb_ids": x["ground_truth_imdb_ids"],
        "assigned_trait": x["assigned_trait"],  # Keep trait for reference
    })

    # Split dataset: test_size samples for testing, rest for training
    dataset = dataset.train_test_split(test_size=test_size, seed=seed)

    print(f"Training samples: {len(dataset['train'])}")
    print(f"Evaluation samples: {len(dataset['test'])}")

    return dataset


def load_eval_dataset(num_samples=None, test_size=300, seed=3407):
    """
    Load evaluation dataset with optional sample limiting.

    Args:
        num_samples: Optional limit on number of samples to use (default: None = use all)
        test_size: Number of samples in test set (default: 300)
        seed: Random seed for reproducible splitting (default: 3407)

    Returns:
        The test split of the dataset, optionally limited to num_samples
    """
    dataset = load_and_prepare_dataset(test_size=test_size, seed=seed)
    eval_dataset = dataset["test"]

    # Limit samples if specified
    if num_samples is not None and num_samples < len(eval_dataset):
        eval_dataset = eval_dataset.select(range(num_samples))
        print(f"Limited to {num_samples} samples")

    return eval_dataset


def load_train_dataset(test_size=300, seed=3407):
    """
    Load training dataset.

    Args:
        test_size: Number of samples to reserve for testing (default: 300)
        seed: Random seed for reproducible splitting (default: 3407)

    Returns:
        The train split of the dataset
    """
    dataset = load_and_prepare_dataset(test_size=test_size, seed=seed)
    return dataset["train"]
