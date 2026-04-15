"""
Retrieval helpers: load pre-computed deterministic rankings and normalise scores
for downstream Plackett-Luce or MMR re-ranking.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)


def load_retrieval_results(
    generator_name: str,
    ranker: str,
    lamp_num: int,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Load pre-computed deterministic retrieval results produced by
    ``retrieval/rank_profiles.py`` or ``retrieval/gold_retriever.py``.

    Returns
    -------
    dict  ``{qid: [(pid, score), ...]}`` sorted by score descending.
    """
    fp = os.path.join(
        ROOT, "retrieval", "retrieval_results",
        generator_name, ranker, f"{lamp_num}.json",
    )
    if not os.path.exists(fp):
        raise FileNotFoundError(
            f"Retrieval results not found: {fp}\n"
            f"Please run first:\n"
            f"  python retrieval/rank_profiles.py "
            f"--ranker {ranker} --generator_name {generator_name} --lamp_num {lamp_num}"
        )
    with open(fp, encoding="utf-8") as fh:
        return json.load(fh)


def normalize_scores_for_pl(scores: np.ndarray, ranker: str) -> np.ndarray:
    """
    Normalise raw retrieval scores to the ``[1, 2]`` range expected by
    Plackett-Luce alpha-temperature exponentiation.

    Special case for the gold (oracle) retriever: binary {0, 1} scores are
    amplified to {0, 10} so the alpha parameter has a visible effect even at
    low alpha values.
    """
    if ranker == "gold":
        return np.where(scores > 0, 10.0, 0.0)

    mn = scores.min()
    if mn < 0:
        scores = scores - mn
        mn = 0.0
    mx = scores.max()
    if mx == mn:
        return np.ones_like(scores, dtype=np.float64)
    normed = (scores - mn) / (mx - mn)   # [0, 1]
    return normed + 1.0                   # [1, 2]


def scores_for_qid(
    retrieval_results: Dict,
    qid: str,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract ordered (scores, pids) arrays for a single query.

    Returns
    -------
    scores  : float64 ndarray, shape (n_candidates,)
    pids    : list[str], same order as scores
    """
    pairs = retrieval_results[qid]
    pids = [p[0] for p in pairs]
    scores = np.array([float(p[1]) for p in pairs], dtype=np.float64)
    return scores, pids
