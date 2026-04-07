"""
Ranking fusion techniques for combining multiple retrieval results.
"""

from typing import List, Dict


def reciprocal_rank_fusion(rank_lists: List[List[str]], k: int = 60) -> List[str]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Args:
        rank_lists: List of ranked document ID lists
        k: Constant to dampen rank impact (default: 60)

    Returns:
        Final fused ranking (list of doc IDs)
    """
    scores: Dict[str, float] = {}

    for rlist in rank_lists:
        for rank, doc_id in enumerate(rlist):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1 / (k + rank)

    # Sort by score (descending)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)