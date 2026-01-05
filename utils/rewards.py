from datasets import load_dataset
import re
import json
from utils.prompts import system_prompt, reasoning_start, reasoning_end, solution_start, solution_end
from utils.movie_mapper import extract_imdb_ids_from_titles

# ==================== Global Configuration ====================
# Load movie sensitivity warnings dataset
print("Loading movie-sensitivity-warnings dataset...")
warnings_dataset = load_dataset("Dionysianspirit/movie-sensitivity-warnings", split="train")

# Create a mapping: imdb_id -> {warning_question: binary_value}
# This will be used to check if a movie contains specific warnings
warnings_mapping = {}
for item in warnings_dataset:
    imdb_id = item['imdb_id']
    # Store all warning columns (exclude imdb_id itself)
    warnings_mapping[imdb_id] = {k: v for k, v in item.items() if k != 'imdb_id'}

print(f"Loaded warnings for {len(warnings_mapping)} movies")



def check_movie_safety(imdb_id, avoid_questions):
    """
    Check if a movie is safe given user's avoid_content questions.
    Returns: 1.0 if safe (no warnings match), 0.0 if unsafe (has warnings)
    """
    if imdb_id not in warnings_mapping:
        # If movie not in warnings database, assume it's safe (or return 0.5 for uncertainty)
        return 0.5

    movie_warnings = warnings_mapping[imdb_id]

    # Check each avoid_content question
    for question in avoid_questions:
        # Normalize the question to match dataset column format
        # The avoid_content questions should match the warning column names
        normalized_q = question.strip()

        # Check if this warning exists for the movie
        if normalized_q in movie_warnings and movie_warnings[normalized_q] == 1:
            # Movie has this warning - NOT SAFE
            return 0.0

    # No warnings found - SAFE
    return 1.0

# ==================== Reward Functions ====================

def calculate_safety_reward(prompts, completions, avoid_content, **kwargs):
    """
    Calculate Safety Reward based on whether recommended movies contain warnings.

    Safety = 0.6 × (Top-5 safety rate) + 0.4 × (Positions 6-20 safety rate)

    Args:
        prompts: List of prompt conversations
        completions: List of model completions
        avoid_content: List of user's avoid_content questions (from dataset)

    Returns:
        List of safety scores (0.0 to 1.0)
    """
    scores = []

    for i, completion in enumerate(completions):
        response = completion[0]["content"]
        avoid_questions = avoid_content[i] if isinstance(avoid_content, list) else avoid_content

        # Extract titles and map to IMDB IDs using fuzzy matching
        imdb_ids, _, _ = extract_imdb_ids_from_titles(response)

        if len(imdb_ids) == 0:
            # No valid recommendations found - penalize
            scores.append(0.0)
            continue

        # Calculate safety for Top-5 positions
        top5_ids = imdb_ids[:5]
        top5_safe_count = sum(check_movie_safety(imdb_id, avoid_questions) for imdb_id in top5_ids)
        top5_safety_rate = top5_safe_count / len(top5_ids) if len(top5_ids) > 0 else 0.0

        # Calculate safety for positions 6-20
        rest_ids = imdb_ids[5:20]
        rest_safe_count = sum(check_movie_safety(imdb_id, avoid_questions) for imdb_id in rest_ids)
        rest_safety_rate = rest_safe_count / len(rest_ids) if len(rest_ids) > 0 else top5_safety_rate

        # Combined safety score with weights
        safety_score = 0.6 * top5_safety_rate + 0.4 * rest_safety_rate
        scores.append(safety_score)

    return scores


def calculate_accuracy_reward(prompts, completions, ground_truth_imdb_ids, **kwargs):
    """
    Calculate Accuracy Reward based on GT hit rate.

    Accuracy = Number of GT hits / Total GT count

    Args:
        prompts: List of prompt conversations
        completions: List of model completions
        ground_truth_imdb_ids: List of ground truth IMDB IDs (from dataset)

    Returns:
        List of accuracy scores (0.0 to 1.0)
    """
    scores = []

    for i, completion in enumerate(completions):
        response = completion[0]["content"]
        gt_ids = ground_truth_imdb_ids[i] if isinstance(ground_truth_imdb_ids, list) else ground_truth_imdb_ids

        if len(gt_ids) == 0:
            # No ground truth - neutral score
            scores.append(0.5)
            continue

        # Extract titles and map to IMDB IDs using fuzzy matching
        predicted_ids, _, _ = extract_imdb_ids_from_titles(response)

        # Count how many GT movies are in predictions
        gt_set = set(gt_ids)
        pred_set = set(predicted_ids)
        hits = len(gt_set.intersection(pred_set))

        accuracy = hits / len(gt_ids)
        scores.append(accuracy)

    return scores


def calculate_coverage_reward(prompts, completions, **kwargs):
    """
    Calculate Coverage Reward based on valid IMDB ID rate.

    Coverage = Number of valid IMDB IDs / Total recommendations

    Args:
        prompts: List of prompt conversations
        completions: List of model completions

    Returns:
        List of coverage scores (0.0 to 1.0)
    """
    scores = []

    for completion in completions:
        response = completion[0]["content"]

        # Extract titles and map to IMDB IDs using fuzzy matching
        imdb_ids, _, _ = extract_imdb_ids_from_titles(response)

        if len(imdb_ids) == 0:
            # No recommendations found - penalize
            scores.append(0.0)
            continue

        # Check how many are valid (exist in our warnings database)
        valid_count = sum(1 for imdb_id in imdb_ids if imdb_id in warnings_mapping)
        coverage = valid_count / len(imdb_ids)
        scores.append(coverage)

    return scores


def calculate_final_reward(prompts, completions, avoid_content, ground_truth_imdb_ids, **kwargs):
    """
    Calculate the final combined reward with weighted components.

    Final Reward = 0.7 × Safety + 0.2 × Accuracy + 0.1 × Coverage

    Args:
        prompts: List of prompt conversations
        completions: List of model completions
        avoid_content: List of user's avoid_content questions
        ground_truth_imdb_ids: List of ground truth IMDB IDs

    Returns:
        List of final reward scores
    """
    # Calculate individual components
    safety_scores = calculate_safety_reward(prompts, completions, avoid_content, **kwargs)
    accuracy_scores = calculate_accuracy_reward(prompts, completions, ground_truth_imdb_ids, **kwargs)
    coverage_scores = calculate_coverage_reward(prompts, completions, **kwargs)

    # Combine with weights: 0.7 × Safety + 0.2 × Accuracy + 0.1 × Coverage
    final_scores = []
    for safety, accuracy, coverage in zip(safety_scores, accuracy_scores, coverage_scores):
        final_score = 0.7 * safety + 0.2 * accuracy + 0.1 * coverage
        # Scale to larger reward range (optional, for better training signal)
        final_score = final_score * 10.0  # Scale from [0,1] to [0,10]
        final_scores.append(final_score)

    return final_scores


# ==================== Debug/Logging Function ====================

global PRINTED_TIMES
PRINTED_TIMES = 0
global PRINT_EVERY_STEPS
PRINT_EVERY_STEPS = 5

def log_reward_details(prompts, completions, avoid_content, ground_truth_imdb_ids, **kwargs):
    """
    Print detailed reward breakdown for debugging.
    This function doesn't return scores, just logs information.
    """
    global PRINTED_TIMES
    global PRINT_EVERY_STEPS

    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        # Calculate components
        safety_scores = calculate_safety_reward(prompts, completions, avoid_content, **kwargs)
        accuracy_scores = calculate_accuracy_reward(prompts, completions, ground_truth_imdb_ids, **kwargs)
        coverage_scores = calculate_coverage_reward(prompts, completions, **kwargs)
        final_scores = calculate_final_reward(prompts, completions, avoid_content, ground_truth_imdb_ids, **kwargs)

        # Print first example
        print('='*80)
        print(f"REWARD BREAKDOWN (Step {PRINTED_TIMES})")
        print('='*80)
        print(f"Prompt: {prompts[0][-1]['content'][:200]}...")
        print(f"\nResponse: {completions[0][0]['content'][:300]}...")
        print(f"\nGround Truth IDs: {ground_truth_imdb_ids[0] if isinstance(ground_truth_imdb_ids, list) else ground_truth_imdb_ids}")
        print(f"Avoid Content: {avoid_content[0][:3] if isinstance(avoid_content, list) else avoid_content[:3]}...")
        print(f"\n--- Scores ---")
        print(f"Safety:   {safety_scores[0]:.4f}")
        print(f"Accuracy: {accuracy_scores[0]:.4f}")
        print(f"Coverage: {coverage_scores[0]:.4f}")
        print(f"Final:    {final_scores[0]:.4f}")
        print('='*80)

    PRINTED_TIMES += 1

    # Return zeros since this is just for logging
    return [0.0] * len(completions)
