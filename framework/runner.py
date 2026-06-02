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

import datetime
import os
import random
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)

from generator.lm import PromptLM
from generator.lm_distributed_inference import PromptLMDistributedInference
from utils import trim_sentence_by_token_len

from framework.artifacts import ArtifactStore, make_run_dir, run_dir_exists
from framework.config import RunConfig, setting_id as make_setting_id
from framework.dataset import DatasetHandler, make_dataset
from framework.metrics import (
    compute_diversity,
    compute_ee,
    compute_eu_for_answer,
    profile_to_text,
    prompt_hash,
)
from framework.reranking import (
    RetrievalList,
    generate_deterministic_list,
    generate_mmr_list,
    generate_pl_lists,
    generate_pl_mmr_lists,
)
from framework.retrieval import load_retrieval_results


class ExperimentRunner:
    """Stateful runner for one experiment configuration."""

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        self._sid = make_setting_id(cfg)

    def run(self) -> ArtifactStore:
        cfg = self.cfg
        self._seed_everything()

        if cfg.resume and cfg.run_id and not run_dir_exists(cfg.run_id):
            raise FileNotFoundError(
                f"Cannot resume run_id '{cfg.run_id}': run directory does not exist."
            )

        dataset: DatasetHandler = make_dataset(cfg)
        expected_queries = dataset.total_queries()
        retrieval_results: Dict = load_retrieval_results(
            generator_name=cfg.generation.generator_name,
            ranker=cfg.retrieval.ranker,
            lamp_num=cfg.dataset.lamp_num,
        )

        run_id = self._make_run_id()
        run_dir = make_run_dir(run_id)
        store = ArtifactStore(run_dir)
        existing_manifest = store.get_manifest()
        store.update_manifest(
            run_id=run_id,
            setting_id=self._sid,
            config=cfg.to_dict(),
            status="running",
            seed=cfg.rerank.seed,
            report_every_queries=cfg.checkpoint.report_every_queries,
            started_at=existing_manifest.get("started_at")
            or datetime.datetime.now().isoformat(timespec="seconds"),
            resumed_at=datetime.datetime.now().isoformat(timespec="seconds") if cfg.resume else None,
            expected_queries=expected_queries,
            n_queries_completed=existing_manifest.get("n_queries_completed", 0),
            last_completed_qid=existing_manifest.get("last_completed_qid"),
            last_seen_query_index=existing_manifest.get("last_seen_query_index", 0),
        )
        store.flush_manifest()
        print(f"Setting: {self._sid}")

        llm = self._make_llm()
        tokenizer = getattr(llm, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("PromptLM must expose a tokenizer for prompt truncation")
        tok_max_len = getattr(llm, "model_max_length", tokenizer.model_max_length)

        metric_name, metric_fn = dataset.get_metric_fn()
        aip_func = dataset.get_aip_func()
        rel_fp = dataset.relevance_mapping_path()

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
        units_since_flush = 0
        n_queries_seen = 0

        for qid, question, target, all_profiles in dataset.iter_queries():
            n_queries_seen += 1
            ret_for_qid = retrieval_results.get(qid, [])
            if not ret_for_qid:
                print("[WARN] Missing retrieval results for one query; skipping.")
                continue

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

            for rl in retrieval_lists:
                unit_key = f"{qid}::{rl.list_id}"
                if unit_key in completed_units:
                    continue

                top_profiles = dataset.find_profiles_by_pids(qid, rl.doc_ids)
                raw_prompt = aip_func(question=question, profiles=top_profiles)
                final_prompt = trim_sentence_by_token_len(
                    raw_prompt, tokenizer=tokenizer, max_tok_len=tok_max_len
                )

                t0 = time.time()
                answer = llm.answer_question(final_prompt=final_prompt).strip()
                elapsed = time.time() - t0

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

                eu_score: Optional[float] = None
                if cfg.metrics.compute_eu and target:
                    eu_score = compute_eu_for_answer(answer, target, metric_fn)

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
                    store.update_manifest(
                        n_queries_completed=len(summarized_qids),
                        last_completed_qid=qid,
                        last_seen_query_index=n_queries_seen,
                    )

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

        store.update_manifest(
            status="completed",
            n_queries_completed=len(summarized_qids),
            last_seen_query_index=n_queries_seen,
        )
        store.flush_manifest()
        summary_fp = store.write_summary()
        macro_fp = store.write_macro_summary()
        print(f"Completed setting: {self._sid}")
        print(f"Summary: {summary_fp}")
        print(f"Macro summary: {macro_fp}")
        return store

    def _make_run_id(self) -> str:
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

        if rr.method == "pl_mmr":
            pids_in_order = [p[0] for p in ret_for_qid]
            profiles_in_order = dataset.find_profiles_by_pids(qid, pids_in_order)
            return generate_pl_mmr_lists(
                retrieval_results_for_qid=ret_for_qid,
                profiles_for_qid=profiles_in_order,
                ranker=cfg.retrieval.ranker,
                pl_alpha=rr.pl_alpha,
                pl_mmr_lambda=rr.mmr_lambda,
                pl_samples=rr.pl_samples,
                top_k=cfg.retrieval.top_k,
                seed=rr.seed,
                qid=qid,
            )

        return [
            generate_deterministic_list(
                retrieval_results_for_qid=ret_for_qid,
                top_k=cfg.retrieval.top_k,
                qid=qid,
            )
        ]

    @staticmethod
    def _reload_retrieval_lists(store: ArtifactStore, qid: str) -> List[RetrievalList]:
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
