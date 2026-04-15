"""
Typed configuration dataclasses for the Fair-RAG experimentation framework.

Canonical IDs
-------------
setting_id  — unique string encoding all hyper-parameters that define one experiment
              condition; used as the folder name under experiment_runs/.
list_id     — per-query retrieval list identifier:
                PL:          "{qid}__pl_s{sample_idx:03d}"
                MMR:         "{qid}__mmr"
                Deterministic: "{qid}__det"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    dataset_type: str = "lamp"          # "lamp" | "trec_rag" (future)
    lamp_num: int = 4                   # LaMP task 1-7
    lamp_split_type: str = "user"       # "user" | "time"
    num_queries: Optional[int] = 50     # None = process all available queries


@dataclass
class RetrievalConfig:
    ranker: str = "bm25"                # "bm25" | "splade" | "contriever" | "gold"
    top_k: int = 5                      # top-K documents passed to LLM context


@dataclass
class RerankConfig:
    method: str = "pl"                  # "pl" (Plackett-Luce) | "mmr" | "deterministic"
    # Plackett-Luce params
    pl_alpha: int = 1                   # temperature α; higher = more deterministic
    pl_samples: int = 10                # N stochastic samples per query
    # MMR params
    mmr_lambda: float = 0.55            # relevance/diversity trade-off (0=max diversity, 1=pure relevance)
    # Reproducibility
    seed: int = 42


@dataclass
class GenerationConfig:
    generator_name: str = "flanT5Small" # model nickname; see utils.models_info
    multi_gpu: bool = False


@dataclass
class MetricsConfig:
    compute_ee: bool = True             # Expected Exposure (EE-D, EE-R, EE-L)
    compute_eu: bool = True             # Expected Utility (per-task metric)
    compute_diversity: bool = True      # Intra-List Diversity + Jaccard


@dataclass
class CheckpointConfig:
    flush_every: int = 1                # flush manifest after every N completed units
    report_every_queries: int = 20      # print rolling averages every N completed queries


@dataclass
class RunConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    resume: bool = False
    run_id: Optional[str] = None        # auto-generated from timestamp + setting_id if None
    run_dir: Optional[str] = None       # override output directory if set

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# ID builders
# ---------------------------------------------------------------------------

def setting_id(cfg: RunConfig) -> str:
    """
    Canonical string that uniquely identifies a set of experiment conditions.
    Example: ``lamp4_user__flanT5Small__bm25__pl_a1_s10__k5__nq50__seed42``
    """
    d = cfg.dataset
    r = cfg.retrieval
    rr = cfg.rerank
    g = cfg.generation

    if rr.method == "pl":
        rerank_str = f"pl_a{rr.pl_alpha}_s{rr.pl_samples}"
    elif rr.method == "mmr":
        lambda_str = str(rr.mmr_lambda).replace(".", "")
        rerank_str = f"mmr_l{lambda_str}"
    else:
        rerank_str = "det"

    nq = f"nq{d.num_queries}" if d.num_queries is not None else "nqall"
    return (
        f"{d.dataset_type}{d.lamp_num}_{d.lamp_split_type}"
        f"__{g.generator_name}__{r.ranker}__{rerank_str}__k{r.top_k}__{nq}"
        f"__seed{rr.seed}"
    )


def list_id_for_pl(qid: str, sample_idx: int) -> str:
    """Retrieval list ID for the i-th Plackett-Luce sample."""
    return f"{qid}__pl_s{sample_idx:03d}"


def list_id_for_mmr(qid: str) -> str:
    """Retrieval list ID for a deterministic MMR reranking."""
    return f"{qid}__mmr"


def list_id_for_deterministic(qid: str) -> str:
    """Retrieval list ID for the plain deterministic base ranking."""
    return f"{qid}__det"
