"""
Dataset abstraction layer.

``DatasetHandler`` is the abstract interface all dataset adapters must implement.
``LaMPDataset`` wraps the existing ``LaMPHandler`` for the LaMP benchmark.
Future adapters (e.g. ``TrecDataset``) can be added without changing the rest of
the framework.
"""

from __future__ import annotations

import json
import os
import sys
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterator, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)

from data.lamp_handler import LaMPHandler
from utils import models_info


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class DatasetHandler(ABC):
    """Contract that every dataset adapter must satisfy."""

    @abstractmethod
    def iter_queries(self) -> Iterator[Tuple[str, str, str, List[Dict]]]:
        """
        Iterate over queries in dataset order, respecting ``num_queries`` limit.

        Yields
        ------
        qid          : str   – query / user identifier
        question     : str   – raw input text
        target       : str   – gold output for EU evaluation
        profiles     : list  – full list of profile dicts (all candidates)
        """

    @abstractmethod
    def relevance_mapping_path(self) -> str:
        """Absolute path to the TSV relevance mapping file used by EE evaluation."""

    @abstractmethod
    def get_aip_func(self) -> Callable:
        """Return the aggregated-input-prompt builder: ``(question, profiles) -> str``."""

    @abstractmethod
    def find_profiles_by_pids(self, qid: str, pids: List[str]) -> List[Dict]:
        """Return ordered list of profile dicts for the given PIDs (preserves PID order)."""

    @abstractmethod
    def get_metric_fn(self) -> Tuple[str, Callable]:
        """Return ``(metric_name, metric_fn)`` for EU computation."""


# ---------------------------------------------------------------------------
# LaMP adapter
# ---------------------------------------------------------------------------

_LAMP_LABELS: Dict[int, List[str]] = {
    1: ["[1]", "[2]"],
    2: [
        "sci-fi", "based on a book", "comedy", "action", "twist ending",
        "dystopia", "dark comedy", "classic", "psychology", "fantasy",
        "romance", "thought-provoking", "social commentary", "violence", "true story",
    ],
    3: ["1", "2", "3", "4", "5"],
}


class LaMPDataset(DatasetHandler):
    """
    LaMP dataset adapter.

    Wraps ``LaMPHandler`` and pre-loads all profiles into memory so
    ``find_profiles_by_pids`` is O(1) instead of re-reading the JSON file on
    every call.
    """

    def __init__(
        self,
        dataset_config,   # DatasetConfig
        retrieval_config, # RetrievalConfig
        generation_config # GenerationConfig
    ) -> None:
        self.lamp_num = dataset_config.lamp_num
        self.split_type = dataset_config.lamp_split_type
        self.num_queries = dataset_config.num_queries
        self.generator_name = generation_config.generator_name
        self.top_k = retrieval_config.top_k
        self.lamp_dir = f"lamp_utility_labels_{self.generator_name}"

        self._handler = LaMPHandler(
            lamp_dir_name=self.lamp_dir,
            split_type=self.split_type,
            tokenizer_model_name=models_info[self.generator_name]["model_id"],
            k=self.top_k,
        )

        # Pre-load all data once to avoid redundant disk reads during long runs.
        self._ordered_qids: List[str] = []
        self._inputs: Dict[str, Dict] = {}
        self._outputs: Dict[str, Dict] = {}
        self._profile_cache: Dict[str, Dict[str, Dict]] = {}
        self._preload()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _preload(self) -> None:
        raw_inputs = list(self._handler.get_inputs_file_iterator(self.lamp_num))
        raw_outputs = list(self._handler.get_outputs_file_iterator(self.lamp_num))

        self._ordered_qids = [e["id"] for e in raw_inputs]
        self._inputs = {e["id"]: e for e in raw_inputs}
        self._outputs = {e["id"]: e for e in raw_outputs}
        # qid → {pid → profile_dict}
        self._profile_cache = {
            e["id"]: {p["id"]: p for p in e["profile"]}
            for e in raw_inputs
        }

    # ------------------------------------------------------------------
    # DatasetHandler interface
    # ------------------------------------------------------------------

    def iter_queries(self):
        count = 0
        for qid in self._ordered_qids:
            if self.num_queries is not None and count >= self.num_queries:
                break
            entry = self._inputs[qid]
            question = entry["input"]
            target = self._outputs[qid]["output"]
            profiles = entry["profile"]
            yield qid, question, target, profiles
            count += 1

    def relevance_mapping_path(self) -> str:
        return os.path.join(
            ROOT, "data", self.lamp_dir,
            f"{self.lamp_num}_relevance_mapping.tsv",
        )

    def get_aip_func(self) -> Callable:
        return self._handler.get_aip_func(self.lamp_num)

    def find_profiles_by_pids(self, qid: str, pids: List[str]) -> List[Dict]:
        pid_map = self._profile_cache.get(qid, {})
        return [pid_map[pid] for pid in pids if pid in pid_map]

    def get_metric_fn(self) -> Tuple[str, Callable]:
        from eval.lamp_metrics import (
            get_metric_fn_accuracy,
            get_metric_fn_mae,
            get_metric_fn_rouge_L,
        )
        if self.lamp_num in {1, 2}:
            return "acc", get_metric_fn_accuracy(_LAMP_LABELS[self.lamp_num])
        elif self.lamp_num == 3:
            return "mae", get_metric_fn_mae()
        else:
            return "rouge-l", get_metric_fn_rouge_L()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_dataset(cfg) -> DatasetHandler:
    """Factory: return the right DatasetHandler for the given RunConfig."""
    if cfg.dataset.dataset_type == "lamp":
        return LaMPDataset(cfg.dataset, cfg.retrieval, cfg.generation)
    raise NotImplementedError(
        f"dataset_type='{cfg.dataset.dataset_type}' is not yet implemented. "
        "Add a new DatasetHandler subclass and register it in make_dataset()."
    )
