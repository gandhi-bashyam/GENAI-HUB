"""
Normalization utilities for ranking scores.

Used to bring different scoring systems (BM25, vector similarity)
onto a comparable scale.
"""

import math
from typing import List


def z_score_normalize(scores: List[float]) -> List[float]:
    """
    Apply Z-score normalization.

    Args:
        scores: List of raw scores

    Returns:
        List of normalized scores (mean=0, std=1)
    """
    if not scores:
        return []

    mean = sum(scores) / len(scores)
    variance = sum((x - mean) ** 2 for x in scores) / len(scores)
    std = variance ** 0.5

    if std == 0:
        return [0.0 for _ in scores]

    return [(x - mean) / std for x in scores]


def softmax(scores: List[float]) -> List[float]:
    """
    Convert scores into probabilities.

    Args:
        scores: List of raw scores

    Returns:
        Probability distribution (sum = 1)
    """
    if not scores:
        return []

    # Stability trick (VERY IMPORTANT in real systems)
    max_score = max(scores)
    exp_scores = [math.exp(x - max_score) for x in scores]

    total = sum(exp_scores)
    if total == 0:
        return [0.0 for _ in scores]

    return [x / total for x in exp_scores]