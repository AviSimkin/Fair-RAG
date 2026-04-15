"""
ExperimentRunner — orchestrates a full Fair-RAG evaluation run.

Flow per query
--------------
1. Generate (or reload from disk) ranked retrieval lists.
2. Compute Expected Exposure (EE-D, EE-R, EE-L) using ALL lists together.
3. For each pending (qid, list_id) unit:
   a. Build LLM prompt from selected profiles.
   b. Query the LLM.
   c. Compute Expected Utility (EU) against gold label.
   d. Compute diversity metrics (ILD-Jaccard, mean Jaccard).
   e. Persist all artifacts atomically.

Resume logic
------------
- Retrieval lists recorded in ``retrieval_lists.jsonl`` are reloaded verbatim
  so stochastic PL samples are stable across resumed runs.
- EE is skipped for qids already in ``ee_metrics.jsonl``.
- LLM+EU+diversity are skipped for units already in ``per_list_metrics.jsonl``.
"""

from __future__ import annotations

import os
import random
import sys
import time
import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)

from transformers import AutoTokenizer

from utils import models_info, trim_sentence_by_token_len
from generator.lm import PromptLM
from generator.lm_distributed_inference import PromptLMDistributedInference

from framework.config import RunConfig, setting_id as make_setting_id
from framework.dataset import make_dataset, DatasetHandler
from framework.retrieval import load_retrieval_results
from framework.reranking import (
    RetrievalList,
    generate_pl_lists,
    generate_mmr_list,
    generate_deterministic_list,
)
from framework.artifacts import ArtifactStore, make_run_dir
from framework.metrics import (
    compute_ee,
    compute_eu_for_answer,
    compute_diversity,
    profile_to_text,
    prompt_hash,
)
from hf_runtime import from_pretrained_kwargs


class ExperimentRunner:
    """
    Stateful runner for one experiment configuration.

    Usage
    -----
    ::
        cfg = RunConfig(...)
        runner = ExperimentRunner(cfg)
        store  = runner.run()   # blocks; returns ArtifactStore on completion
    """

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        self._sid = make_setting_id(cfg)

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def run(self) -> ArtifactStore:
        cfg = self.cfg
        self._seed_everything()

        # --- 1. Setup artefact store ---------------------------------
        run_id = self._make_run_id()
        run_dir = make_run_dir(run_id)
        store = ArtifactStore(run_dir)
        store.update_manifest(
            run_id=run_id,
            setting_id=self._sid,
            config=cfg.to_dict(),
            status="running",
            seed=cfg.rerank.seed,
            report_every_queries=cfg.checkpoint.report_every_queries,
            started_at=datetime.datetime.now().isoformat(timespec="seconds"),
        )
        store.flush_manifest()
        print(f"Setting: {self._sid}")

        # --- 2. Load dataset & retrieval results ---------------------
        dataset: DatasetHandler = make_dataset(cfg)
        retrieval_results: Dict = load_retrieval_results(
            generator_name=cfg.generation.generator_name,
            ranker=cfg.retrieval.ranker,
            lamp_num=cfg.dataset.lamp_num,
        )

        # --- 3. Load LLM & tokenizer ---------------------------------
        llm = self._make_llm()
        model_id = models_info[cfg.generation.generator_name]["model_id"]
        tokenizer = AutoTokenizer.from_pretrained(model_id, **from_pretrained_kwargs())
        tok_max_len = tokenizer.model_max_length

        # --- 4. Metric function & eval handles -----------------------
        metric_name, metric_fn = dataset.get_metric_fn()
        aip_func = dataset.get_aip_func()
        rel_fp = dataset.relevance_mapping_path()

        # --- 5. Resume state  ----------------------------------------
        if cfg.resume:
            completed_units: Set[str] = store.get_completed_answer_units()
            ee_done_qids: Set[str] = store.get_ee_completed_qids()
            saved_retrieval_qids: Set[str] = store.get_saved_retrieval_qids()
            summarized_qids: Set[str] = store.get_query_summary_qids()
            print(f"Resuming run: {len(summarized_qids)} summarized queries already completed")
        else:
            completed_units = set()
            ee_done_qids = set()
            saved_retrieval_qids = set()
            summarized_qids = set()

        query_summaries = store.load_query_summaries()
        metric_totals = self._init_metric_totals(query_summaries)
        report_interval = max(1, cfg.checkpoint.report_every_queries)
        queries_since_report = 0

        # --- 6. Main loop  -------------------------------------------
        units_since_flush = 0
        n_queries_seen = 0

        for qid, question, target, all_profiles in dataset.iter_queries():
            n_queries_seen += 1
            ret_for_qid = retrieval_results.get(qid, [])
            if not ret_for_qid:
                print("[WARN] Missing retrieval results for one query; skipping.")
                continue

            # 6a. Retrieval lists (generate or reload) ----------------
            if cfg.resume and qid in saved_retrieval_qids:
                retrieval_lists = self._reload_retrieval_lists(store, qid)
            else:
                retrieval_lists = self._generate_lists(
                    qid=qid,
                    ret_for_qid=ret_for_qid,
                    dataset=dataset,
                    all_profiles=all_profiles,
                )
                for rl in retrieval_lists:
                    store.append_retrieval_list(
                        qid=rl.qid,
                        list_id=rl.list_id,
                        doc_ids=rl.doc_ids,
                        det_indices=rl.det_indices,
                        method=rl.method,
                        param=rl.param,
                        sample_idx=rl.sample_idx,
                    )

            # 6b. Expected Exposure -----------------------------------
            if cfg.metrics.compute_ee and qid not in ee_done_qids:
                det_indices_per_list = [rl.det_indices for rl in retrieval_lists]
                ee_res = compute_ee(
                    qid=qid,
                    det_indices_per_list=det_indices_per_list,
                    retrieval_results_for_qid=ret_for_qid,
                    rel_mapping_fp=rel_fp,
                    top_k=cfg.retrieval.top_k,
                )
                store.append_ee_metrics(
                    qid=qid,
                    ee_disparity=ee_res["ee_disparity"],
                    ee_relevance=ee_res["ee_relevance"],
                    ee_difference=ee_res["ee_difference"],
                )
                ee_done_qids.add(qid)

            # 6c. Per-list: LLM + EU + diversity ----------------------
            for rl in retrieval_lists:
                unit_key = f"{qid}::{rl.list_id}"
                if unit_key in completed_units:
                    continue

                # Build prompt
                top_profiles = dataset.find_profiles_by_pids(qid, rl.doc_ids)
                raw_prompt = aip_func(question=question, profiles=top_profiles)
                final_prompt = trim_sentence_by_token_len(
                    raw_prompt, tokenizer=tokenizer, max_tok_len=tok_max_len
                )

                # LLM inference
                t0 = time.time()
                answer = llm.answer_question(final_prompt=final_prompt).strip()
                elapsed = time.time() - t0

                # Guard against empty/whitespace-only predictions
                if answer == "" or all(c in {".", " "} for c in answer):
                    answer = "<empty>"

                store.append_llm_answer(
                    qid=qid,
                    list_id=rl.list_id,
                    setting_id=self._sid,
                    top_k=cfg.retrieval.top_k,
                    prompt_hash=prompt_hash(final_prompt),
                    answer=answer,
                    elapsed_s=elapsed,
                )

                # EU
                eu_score: Optional[float] = None
                if cfg.metrics.compute_eu and target:
                    eu_score = compute_eu_for_answer(answer, target, metric_fn)

                # Diversity
                ild_val: Optional[float] = None
                jac_val: Optional[float] = None
                if cfg.metrics.compute_diversity and top_profiles:
                    doc_texts = [profile_to_text(p) for p in top_profiles]
                    div = compute_diversity(doc_texts)
                    ild_val = div["ild_jaccard"]
                    jac_val = div["jaccard_mean"]

                store.append_per_list_metrics(
                    qid=qid,
                    list_id=rl.list_id,
                    setting_id=self._sid,
                    eu_score=eu_score,
                    metric_name=metric_name,
                    ild_jaccard=ild_val,
                    jaccard_mean=jac_val,
                )
                completed_units.add(unit_key)

                units_since_flush += 1
                if units_since_flush >= cfg.checkpoint.flush_every:
                    store.flush_manifest()
                    units_since_flush = 0

            if qid not in summarized_qids:
                query_summary = store.build_query_summary_for_qid(qid)
                if query_summary is not None:
                    query_summary.update(
                        {
                            "setting_id": self._sid,
                            "run_id": run_id,
                            "seed": cfg.rerank.seed,
                        }
                    )
                    store.append_query_summary(query_summary)
                    summarized_qids.add(qid)
                    self._update_metric_totals(metric_totals, query_summary)
                    queries_since_report += 1
                    store.update_manifest(n_queries_completed=len(summarized_qids))

                    if queries_since_report >= report_interval:
                        progress = self._build_progress_report(
                            metric_totals=metric_totals,
                            query_count=len(summarized_qids),
                            run_id=run_id,
                            setting_id=self._sid,
                        )
                        store.append_progress_report(progress)
                        self._print_progress_report(progress)
                        queries_since_report = 0

        if queries_since_report > 0 and metric_totals["n_queries"] > 0:
            progress = self._build_progress_report(
                metric_totals=metric_totals,
                query_count=len(summarized_qids),
                run_id=run_id,
                setting_id=self._sid,
            )
            store.append_progress_report(progress)
            self._print_progress_report(progress)

        # --- 7. Finalise  -------------------------------------------
        store.update_manifest(status="completed", n_queries_completed=len(summarized_qids))
        store.flush_manifest()
        summary_fp = store.write_summary()
        macro_fp = store.write_macro_summary()
        print(f"Completed setting: {self._sid}")
        print(f"Summary: {summary_fp}")
        print(f"Macro summary: {macro_fp}")
        return store

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_run_id(self) -> str:
        """Generate a unique run identifier: <timestamp>_<setting_id>."""
        if self.cfg.run_id:
            return self.cfg.run_id
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{ts}_{self._sid}"

    def _make_llm(self):
        cfg = self.cfg
        if cfg.generation.multi_gpu:
            return PromptLMDistributedInference(
                model_name=cfg.generation.generator_name,
                seed=cfg.rerank.seed,
            )
        return PromptLM(model_name=cfg.generation.generator_name, seed=cfg.rerank.seed)

    def _seed_everything(self) -> None:
        seed = self.cfg.rerank.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def _init_metric_totals(query_summaries: List[Dict]) -> Dict[str, float]:
        totals = {
            "n_queries": 0,
            "expected_utility": 0.0,
            "ee_disparity": 0.0,
            "ee_relevance": 0.0,
            "ee_difference": 0.0,
            "avg_ild_jaccard": 0.0,
            "avg_jaccard_mean": 0.0,
        }
        for row in query_summaries:
            ExperimentRunner._update_metric_totals(totals, row)
        return totals

    @staticmethod
    def _update_metric_totals(metric_totals: Dict[str, float], query_summary: Dict) -> None:
        metric_totals["n_queries"] += 1
        for key in (
            "expected_utility",
            "ee_disparity",
            "ee_relevance",
            "ee_difference",
            "avg_ild_jaccard",
            "avg_jaccard_mean",
        ):
            value = query_summary.get(key)
            if value is not None:
                metric_totals[key] += value

    @staticmethod
    def _build_progress_report(
        metric_totals: Dict[str, float],
        query_count: int,
        run_id: str,
        setting_id: str,
    ) -> Dict[str, Optional[float]]:
        denom = max(metric_totals["n_queries"], 1)
        return {
            "run_id": run_id,
            "setting_id": setting_id,
            "queries_completed": query_count,
            "avg_expected_utility": metric_totals["expected_utility"] / denom,
            "avg_ee_disparity": metric_totals["ee_disparity"] / denom,
            "avg_ee_relevance": metric_totals["ee_relevance"] / denom,
            "avg_ee_difference": metric_totals["ee_difference"] / denom,
            "avg_ild_jaccard": metric_totals["avg_ild_jaccard"] / denom,
            "avg_jaccard_mean": metric_totals["avg_jaccard_mean"] / denom,
        }

    @staticmethod
    def _print_progress_report(progress: Dict[str, Optional[float]]) -> None:
        print(
            f"Setting: {progress['setting_id']} | "
            f"Average after {progress['queries_completed']} queries: "
            f"EE-D={progress['avg_ee_disparity']:.4f}, "
            f"EE-R={progress['avg_ee_relevance']:.4f}, "
            f"ILD={progress['avg_ild_jaccard']:.4f}, "
            f"EU={progress['avg_expected_utility']:.4f}"
        )

    def _generate_lists(
        self,
        qid: str,
        ret_for_qid: List,
        dataset: DatasetHandler,
        all_profiles: List[Dict],
    ) -> List[RetrievalList]:
        """Dispatch to the configured re-ranking method."""
        cfg = self.cfg
        rr = cfg.rerank

        if rr.method == "pl":
            return generate_pl_lists(
                retrieval_results_for_qid=ret_for_qid,
                ranker=cfg.retrieval.ranker,
                pl_alpha=rr.pl_alpha,
                pl_samples=rr.pl_samples,
                top_k=cfg.retrieval.top_k,
                seed=rr.seed,
                qid=qid,
            )

        if rr.method == "mmr":
            # Need profiles in the same order as the retrieval results
            pids_in_order = [p[0] for p in ret_for_qid]
            profiles_in_order = dataset.find_profiles_by_pids(qid, pids_in_order)
            return [
                generate_mmr_list(
                    retrieval_results_for_qid=ret_for_qid,
                    profiles_for_qid=profiles_in_order,
                    top_k=cfg.retrieval.top_k,
                    mmr_lambda=rr.mmr_lambda,
                    qid=qid,
                )
            ]

        # deterministic
        return [
            generate_deterministic_list(
                retrieval_results_for_qid=ret_for_qid,
                top_k=cfg.retrieval.top_k,
                qid=qid,
            )
        ]

    @staticmethod
    def _reload_retrieval_lists(store: ArtifactStore, qid: str) -> List[RetrievalList]:
        """Reconstruct RetrievalList objects from saved JSONL records."""
        records = store.load_retrieval_lists_for_qid(qid)
        return [
            RetrievalList(
                qid=r["qid"],
                list_id=r["list_id"],
                doc_ids=r["doc_ids"],
                det_indices=r["det_indices"],
                method=r["method"],
                param=r["param"],
                sample_idx=r["sample_idx"],
            )
            for r in records
        ]
