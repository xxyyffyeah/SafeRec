"""
Movie title to IMDB ID mapping utilities.

This module provides functions to:
1. Extract movie titles from model responses
2. Map movie titles to IMDB IDs using the reddit-movie-entity2id dataset
3. Use fuzzy matching with RapidFuzz for better title matching
"""

import re
from datasets import load_dataset
from rapidfuzz import process, fuzz


# ==================== Global Configuration ====================
print("Loading reddit-movie-entity2id dataset...")
entity2id_dataset = load_dataset("Dionysianspirit/reddit-movie-entity2id", split="train")

# Create mappings for title -> imdb_id lookup
title_to_imdb = {}
title_to_imdb_normalized = {}  # Normalized titles (lowercase, no year) for fuzzy matching

for item in entity2id_dataset:
    title = item['title']
    imdb_id = item['imdb_id']

    # Exact match mapping
    title_to_imdb[title] = imdb_id

    # Normalized mapping (remove year, lowercase, strip)
    # e.g., "The Matrix (1999)" -> "the matrix"
    normalized = re.sub(r'\s*\(\d{4}\)\s*$', '', title).lower().strip()
    title_to_imdb_normalized[normalized] = imdb_id

print(f"Loaded {len(title_to_imdb)} movie title mappings")


# ==================== Helper Functions ====================

def extract_movie_titles_from_response(response_text):
    """
    Extract movie titles from model response (new format without IMDB IDs).

    Expected format:
    <SOLUTION>
    1. Movie Title One
    2. Movie Title Two
    3. Movie Title Three
    </SOLUTION>

    Args:
        response_text: Model response text

    Returns:
        list of movie titles
    """
    # Extract content between <SOLUTION> tags
    solution_match = re.search(r'<SOLUTION>(.*?)</SOLUTION>', response_text, re.DOTALL)
    if not solution_match:
        return []

    solution_text = solution_match.group(1).strip()

    # Extract titles from numbered list
    lines = solution_text.split('\n')
    titles = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove leading numbers and punctuation (e.g., "1. ", "2) ", etc.)
        title = re.sub(r'^\d+[\.\)]\s*', '', line)
        if title:
            titles.append(title)

    return titles


def map_title_to_imdb_id(title, fuzzy_threshold=85):
    """
    Map a movie title to its IMDB ID using the loaded entity2id dataset.
    Uses exact matching first, then falls back to fuzzy matching with RapidFuzz.

    Matching strategy:
    1. Exact match (fastest)
    2. Normalized match (lowercase, no year)
    3. Fuzzy match with token_sort_ratio (handles word order variations)

    Args:
        title: Movie title string
        fuzzy_threshold: Minimum similarity score (0-100) for fuzzy matching (default: 85)

    Returns:
        IMDB ID string if found, None otherwise
    """
    # Try exact match first (fastest)
    if title in title_to_imdb:
        return title_to_imdb[title]

    # Try normalized match (lowercase, no year)
    normalized = re.sub(r'\s*\(\d{4}\)\s*$', '', title).lower().strip()
    if normalized in title_to_imdb_normalized:
        return title_to_imdb_normalized[normalized]

    # Fall back to fuzzy matching using RapidFuzz
    # Use token_sort_ratio for better handling of word order and common variations
    valid_titles = list(title_to_imdb_normalized.keys())

    # Filter out very short titles to avoid false matches
    # (e.g., "god father" shouldn't match "d")
    min_length = max(3, len(normalized) // 2)
    filtered_titles = [t for t in valid_titles if len(t) >= min_length]

    if not filtered_titles:
        return None

    result = process.extractOne(
        normalized,
        filtered_titles,
        scorer=fuzz.token_sort_ratio
    )

    if result is not None:
        match, score, _ = result  # Unpack without using index
        if score >= fuzzy_threshold:
            # Found a good fuzzy match
            return title_to_imdb_normalized[match]

    # No match found
    return None


def extract_imdb_ids_from_titles(response_text):
    """
    Extract movie titles from response and map them to IMDB IDs.
    This is the main function for new models using title-only format.

    Args:
        response_text: Model response text

    Returns:
        tuple: (list of imdb_ids, list of titles, mapping_stats)
            - imdb_ids: Successfully mapped IMDB IDs
            - titles: All extracted titles
            - mapping_stats: dict with 'mapped' and 'unmapped' counts
    """
    titles = extract_movie_titles_from_response(response_text)
    imdb_ids = []
    unmapped_titles = []

    for title in titles:
        imdb_id = map_title_to_imdb_id(title)
        if imdb_id:
            imdb_ids.append(imdb_id)
        else:
            unmapped_titles.append(title)

    stats = {
        'total_titles': len(titles),
        'mapped': len(imdb_ids),
        'unmapped': len(unmapped_titles),
        'unmapped_titles': unmapped_titles
    }

    return imdb_ids, titles, stats
