"""
Re-ranking methods.

Plackett-Luce (stochastic)
--------------------------
Generates ``N`` randomised rankings via Gumbel-max trick.
Each sample becomes one ``RetrievalList`` with a unique ``list_id``.

MMR — Maximal Marginal Relevance (deterministic)
-------------------------------------------------
Produces exactly **one** ``RetrievalList`` per query.
Uses Jaccard similarity on document token sets for the diversity term.

Algorithm:
    score(d) = λ · rel(d, q) − (1 − λ) · max_{d' ∈ S} sim(d, d')
where
    rel(d, q) — base retrieval score normalised to [0, 1]
    S         — already selected documents
    λ         — trade-off (default 0.55; higher = more relevance-focused)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)

from perturbation import plackettluce as pl_mod
from framework.retrieval import normalize_scores_for_pl
from framework.config import list_id_for_pl, list_id_for_mmr, list_id_for_deterministic


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class RetrievalList:
    """
    One concrete ranked list of documents for a single query.

    Attributes
    ----------
    qid          : query identifier
    list_id      : unique list identifier (encodes method + sample index)
    doc_ids      : ordered PIDs in list order (length == top_k for MMR/det;
                   full cutoff length for PL samples)
    det_indices  : positions into the original deterministic ranking
                   (same order as doc_ids); used to rebuild TREC top-files for EE
    method       : "pl" | "mmr" | "det"
    param        : human-readable param string e.g. "alpha=2" or "lambda=0.55"
    sample_idx   : index within the method's sample set (0 for deterministic/MMR)
    """
    qid: str
    list_id: str
    doc_ids: List[str]
    det_indices: List[int]
    method: str
    param: str
    sample_idx: int = 0


# ---------------------------------------------------------------------------
# Plackett-Luce
# ---------------------------------------------------------------------------

def generate_pl_lists(
    retrieval_results_for_qid: List,   # [(pid, score), ...]
    ranker: str,
    pl_alpha: int,
    pl_samples: int,
    top_k: int,
    seed: int,
    qid: str,
) -> List[RetrievalList]:
    """
    Generate ``pl_samples`` stochastic rankings via Plackett-Luce sampling.

    Parameters
    ----------
    retrieval_results_for_qid : list of (pid, score) in deterministic rank order
    ranker                    : e.g. "bm25"; used for score normalisation
    pl_alpha                  : temperature exponent; higher = more deterministic
    pl_samples                : number of samples N
    top_k                     : cutoff (list length)
    seed                      : numpy random seed
    qid                       : query id

    Returns
    -------
    List of ``RetrievalList``, one per sample.
    """
    np.random.seed(seed)

    pids = [p[0] for p in retrieval_results_for_qid]
    raw_scores = np.array([float(p[1]) for p in retrieval_results_for_qid], dtype=np.float64)

    normed = normalize_scores_for_pl(raw_scores, ranker)
    scores = normed ** pl_alpha
    cutoff = min(top_k, len(pids))

    rankings, *_ = pl_mod.gumbel_sample_rankings(
        scores, pl_samples, cutoff=cutoff, doc_prob=False
    )

    result: List[RetrievalList] = []
    for i, ranking in enumerate(rankings):
        det_idx = ranking.tolist()          # indices into deterministic ranking
        doc_ids = [pids[j] for j in det_idx]
        result.append(RetrievalList(
            qid=qid,
            list_id=list_id_for_pl(qid, i),
            doc_ids=doc_ids,
            det_indices=det_idx,
            method="pl",
            param=f"alpha={pl_alpha}",
            sample_idx=i,
        ))
    return result


# ---------------------------------------------------------------------------
# MMR helpers
# ---------------------------------------------------------------------------

def _profile_to_tokens(profile: dict) -> frozenset:
    """Extract a token set from a profile dict (all string fields except id/date)."""
    skip = {"id", "date"}
    parts: List[str] = []
    for k, v in profile.items():
        if k not in skip and isinstance(v, str):
            parts.append(v)
    return frozenset(" ".join(parts).lower().split())


def _jaccard_sim(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


# ---------------------------------------------------------------------------
# MMR re-ranker
# ---------------------------------------------------------------------------

def generate_mmr_list(
    retrieval_results_for_qid: List,   # [(pid, score), ...]
    profiles_for_qid: List[Dict],      # full profile dicts, same order as retrieval_results
    top_k: int,
    mmr_lambda: float,
    qid: str,
) -> RetrievalList:
    """
    Deterministic MMR re-ranking.

    Parameters
    ----------
    retrieval_results_for_qid : list of (pid, score) in deterministic rank order
    profiles_for_qid          : profile dicts in the same order as retrieval_results
    top_k                     : desired output list length
    mmr_lambda                : relevance weight (0.55 default)
    qid                       : query id

    Returns
    -------
    Single ``RetrievalList`` with ``method="mmr"``.
    """
    pids = [p[0] for p in retrieval_results_for_qid]
    raw_scores = np.array([float(p[1]) for p in retrieval_results_for_qid], dtype=np.float64)

    # Normalise relevance scores to [0, 1]
    mn, mx = raw_scores.min(), raw_scores.max()
    if mx > mn:
        rel_scores = (raw_scores - mn) / (mx - mn)
    else:
        rel_scores = np.ones_like(raw_scores)

    # Build token sets for each document
    pid_to_tokens: Dict[str, frozenset] = {}
    for pid, prof in zip(pids, profiles_for_qid):
        pid_to_tokens[pid] = _profile_to_tokens(prof)

    # Greedy MMR selection
    selected_indices: List[int] = []
    remaining = list(range(len(pids)))
    n_select = min(top_k, len(pids))

    for _ in range(n_select):
        best_idx: Optional[int] = None
        best_score = -np.inf

        for i in remaining:
            rel = rel_scores[i]
            if selected_indices:
                max_sim = max(
                    _jaccard_sim(pid_to_tokens[pids[i]], pid_to_tokens[pids[j]])
                    for j in selected_indices
                )
            else:
                max_sim = 0.0

            mmr_score = mmr_lambda * rel - (1.0 - mmr_lambda) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    doc_ids = [pids[i] for i in selected_indices]
    return RetrievalList(
        qid=qid,
        list_id=list_id_for_mmr(qid),
        doc_ids=doc_ids,
        det_indices=selected_indices,
        method="mmr",
        param=f"lambda={mmr_lambda}",
        sample_idx=0,
    )


# ---------------------------------------------------------------------------
# Deterministic (passthrough) — kept for completeness and EE baseline
# ---------------------------------------------------------------------------

def generate_deterministic_list(
    retrieval_results_for_qid: List,
    top_k: int,
    qid: str,
) -> RetrievalList:
    """Return the plain deterministic base ranking (no re-ranking)."""
    pids = [p[0] for p in retrieval_results_for_qid]
    cutoff = min(top_k, len(pids))
    det_idx = list(range(cutoff))
    doc_ids = pids[:cutoff]
    return RetrievalList(
        qid=qid,
        list_id=list_id_for_deterministic(qid),
        doc_ids=doc_ids,
        det_indices=det_idx,
        method="det",
        param="",
        sample_idx=0,
    )
