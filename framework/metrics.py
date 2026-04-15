"""
Metrics engine: Expected Exposure (EE-D, EE-R, EE-L), Expected Utility (EU),
and lexical diversity (ILD-Jaccard, mean Jaccard similarity).

All ``compute_*`` functions are stateless; they take raw data and return dicts.
"""

from __future__ import annotations

import os
import sys
import hashlib
import tempfile
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)

from utils import make_trec_top_file_for_single_qid, make_trec_rel_file_for_single_qid
from expected_exposure import expeval

_EE_PARAMS = {
    "umType": "rbp",
    "umPatience": 1,
    "umUtility": 0.5,
    "binarize": False,
    "groupEvaluation": False,
    "complete": False,
    "normalize": True,
    "relfn": "",
    "topfn": "",
}


# ---------------------------------------------------------------------------
# Expected Exposure
# ---------------------------------------------------------------------------

def compute_ee(
    qid: str,
    det_indices_per_list: List[List[int]],   # shape (n_lists, top_k)
    retrieval_results_for_qid: List,          # [(pid, score), ...]
    rel_mapping_fp: str,
    top_k: int,
    remove_temp: bool = True,
) -> Dict[str, Optional[float]]:
    """
    Compute EE-D, EE-R, and EE-L for one query over all its ranked lists.

    Parameters
    ----------
    det_indices_per_list : each row is a list of indices into
                           ``retrieval_results_for_qid``; typically all PL samples
                           (or a single row for MMR / deterministic)
    """
    rankings = np.array(det_indices_per_list, dtype=np.intp)

    try:
        top_fp = make_trec_top_file_for_single_qid(
            qid=qid,
            rankings=rankings,
            retrieval_results=retrieval_results_for_qid,
            run_id="framework_exp",
        )
        rel_fp = make_trec_rel_file_for_single_qid(
            qid=qid,
            relevance_mapping_fp=rel_mapping_fp,
        )
        params = dict(_EE_PARAMS)
        params["topfn"] = top_fp
        params["relfn"] = rel_fp
        result = expeval.run(parameters=params, k=top_k)
    except Exception as exc:
        print(f"[EE] Warning: EE computation failed for qid={qid}: {exc}")
        result = None
    finally:
        if remove_temp:
            for fp in [top_fp, rel_fp]:
                try:
                    os.remove(fp)
                except Exception:
                    pass

    if result is None:
        return {"ee_disparity": None, "ee_relevance": None, "ee_difference": None}
    return {
        "ee_disparity": result.get("disparity"),
        "ee_relevance": result.get("relevance"),
        "ee_difference": result.get("difference"),
    }


# ---------------------------------------------------------------------------
# Expected Utility  (per individual ranking)
# ---------------------------------------------------------------------------

def compute_eu_for_answer(
    prediction: str,
    target: str,
    metric_fn: Callable,
) -> float:
    """
    Compute utility for a single (prediction, target) pair.
    Returns a scalar utility score (metric-dependent).
    Note: MAE is lower-is-better; callers may invert if needed.
    """
    scores: List[float] = metric_fn([prediction], [target])
    return float(scores[0])


# ---------------------------------------------------------------------------
# Diversity: ILD-Jaccard and mean pairwise Jaccard
# ---------------------------------------------------------------------------

def _text_to_tokens(text: str) -> frozenset:
    return frozenset(text.lower().split())


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def compute_diversity(doc_texts: List[str]) -> Dict[str, Optional[float]]:
    """
    Compute intra-list diversity metrics for a single ranked list.

    Returns
    -------
    ild_jaccard  : average pairwise Jaccard **distance** = 1 − Jaccard-similarity
                   (higher = more diverse)
    jaccard_mean : average pairwise Jaccard **similarity**  (lower = more diverse)
    """
    n = len(doc_texts)
    if n < 2:
        return {"ild_jaccard": None, "jaccard_mean": None}

    token_sets = [_text_to_tokens(t) for t in doc_texts]
    sims: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(_jaccard(token_sets[i], token_sets[j]))

    mean_sim = sum(sims) / len(sims)
    return {
        "ild_jaccard": 1.0 - mean_sim,   # ILD = diversity (higher = more diverse)
        "jaccard_mean": mean_sim,         # raw similarity (lower = more diverse)
    }


# ---------------------------------------------------------------------------
# Profile text extractor (shared with reranking module)
# ---------------------------------------------------------------------------

def profile_to_text(profile: dict) -> str:
    """Flatten a profile dict to plain text for diversity computation."""
    skip = {"id", "date"}
    parts: List[str] = []
    for k, v in profile.items():
        if k not in skip and isinstance(v, str):
            parts.append(v)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Prompt hashing (for deduplication in LLM answer cache)
# ---------------------------------------------------------------------------

def prompt_hash(prompt: str) -> str:
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()[:12]
