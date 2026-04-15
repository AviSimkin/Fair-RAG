"""
Crash-safe JSONL artifact store + atomic run manifest.

Layout
------
experiment_runs/
  {run_id}/
    manifest.json          – run metadata; updated atomically via tmp-rename
    retrieval_lists.jsonl  – one JSON line per RetrievalList
    ee_metrics.jsonl       – one JSON line per qid (EE-D, EE-R, EE-L)
    llm_answers.jsonl      – one JSON line per (qid, list_id)
    per_list_metrics.jsonl – one JSON line per (qid, list_id); EU + diversity

Checkpoint/Resume
-----------------
Completed units are identified by scanning the JSONL files on startup rather
than storing them in `manifest.json` (avoids expensive full-manifest rewrites).

``get_completed_answer_units()`` → set of "<qid>::<list_id>" strings
``get_ee_completed_qids()``      → set of qids whose EE has been computed
``get_saved_retrieval_qids()``   → set of qids whose lists are already on disk
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)

_RETRIEVAL_FILE = "retrieval_lists.jsonl"
_EE_FILE = "ee_metrics.jsonl"
_ANSWERS_FILE = "llm_answers.jsonl"
_PER_LIST_FILE = "per_list_metrics.jsonl"
_QUERY_SUMMARY_FILE = "query_summary.jsonl"
_PROGRESS_FILE = "progress_reports.jsonl"
_MANIFEST_FILE = "manifest.json"


class ArtifactStore:
    """
    Append-safe JSONL artifact store for one experiment run.

    Each ``append_*`` call writes a single JSON line followed by a flush+fsync
    so interrupted runs do not lose completed work.
    """

    def __init__(self, run_dir: str) -> None:
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self._manifest: Dict[str, Any] = self._load_manifest()

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _load_manifest(self) -> Dict[str, Any]:
        fp = self._path(_MANIFEST_FILE)
        if os.path.exists(fp):
            with open(fp, encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def flush_manifest(self) -> None:
        """Atomically write the in-memory manifest to disk."""
        fp = self._path(_MANIFEST_FILE)
        tmp = fp + ".tmp"
        self._manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._manifest, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, fp)

    def update_manifest(self, **kwargs: Any) -> None:
        self._manifest.update(kwargs)

    def get_manifest(self) -> Dict[str, Any]:
        return dict(self._manifest)

    # ------------------------------------------------------------------
    # JSONL helpers
    # ------------------------------------------------------------------

    def _path(self, filename: str) -> str:
        return os.path.join(self.run_dir, filename)

    def _append_jsonl(self, filename: str, record: Dict) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with open(self._path(filename), "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

    def load_jsonl(self, filename: str) -> List[Dict]:
        fp = self._path(filename)
        if not os.path.exists(fp):
            return []
        records: List[Dict] = []
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # skip corrupted partial writes from crashes
        return records

    # ------------------------------------------------------------------
    # Typed append methods
    # ------------------------------------------------------------------

    def append_retrieval_list(
        self,
        qid: str,
        list_id: str,
        doc_ids: List[str],
        det_indices: List[int],
        method: str,
        param: str,
        sample_idx: int,
    ) -> None:
        self._append_jsonl(_RETRIEVAL_FILE, {
            "qid": qid,
            "list_id": list_id,
            "doc_ids": doc_ids,
            "det_indices": det_indices,
            "method": method,
            "param": param,
            "sample_idx": sample_idx,
        })

    def append_ee_metrics(
        self,
        qid: str,
        ee_disparity: Optional[float],
        ee_relevance: Optional[float],
        ee_difference: Optional[float],
    ) -> None:
        self._append_jsonl(_EE_FILE, {
            "qid": qid,
            "ee_disparity": ee_disparity,
            "ee_relevance": ee_relevance,
            "ee_difference": ee_difference,
        })

    def append_llm_answer(
        self,
        qid: str,
        list_id: str,
        setting_id: str,
        top_k: int,
        prompt_hash: str,
        answer: str,
        elapsed_s: float,
    ) -> None:
        self._append_jsonl(_ANSWERS_FILE, {
            "qid": qid,
            "list_id": list_id,
            "setting_id": setting_id,
            "top_k": top_k,
            "prompt_hash": prompt_hash,
            "answer": answer,
            "elapsed_s": elapsed_s,
        })

    def append_per_list_metrics(
        self,
        qid: str,
        list_id: str,
        setting_id: str,
        eu_score: Optional[float],
        metric_name: str,
        ild_jaccard: Optional[float],
        jaccard_mean: Optional[float],
    ) -> None:
        self._append_jsonl(_PER_LIST_FILE, {
            "qid": qid,
            "list_id": list_id,
            "setting_id": setting_id,
            "eu_score": eu_score,
            "metric_name": metric_name,
            "ild_jaccard": ild_jaccard,
            "jaccard_mean": jaccard_mean,
        })

    def append_query_summary(self, record: Dict[str, Any]) -> None:
        self._append_jsonl(_QUERY_SUMMARY_FILE, record)

    def append_progress_report(self, record: Dict[str, Any]) -> None:
        self._append_jsonl(_PROGRESS_FILE, record)

    # ------------------------------------------------------------------
    # Resume helpers — reconstruct completed sets from JSONL files
    # ------------------------------------------------------------------

    def get_completed_answer_units(self) -> Set[str]:
        """Return set of "<qid>::<list_id>" strings for units with saved per-list metrics."""
        return {
            f"{r['qid']}::{r['list_id']}"
            for r in self.load_jsonl(_PER_LIST_FILE)
        }

    def get_ee_completed_qids(self) -> Set[str]:
        """Return set of qids whose EE metrics have been saved."""
        return {r["qid"] for r in self.load_jsonl(_EE_FILE)}

    def get_saved_retrieval_qids(self) -> Set[str]:
        """Return set of qids for which retrieval lists are already on disk."""
        return {r["qid"] for r in self.load_jsonl(_RETRIEVAL_FILE)}

    def load_retrieval_lists_for_qid(self, qid: str) -> List[Dict]:
        """Load all retrieval list records saved for a specific qid."""
        return [r for r in self.load_jsonl(_RETRIEVAL_FILE) if r["qid"] == qid]

    def load_answers_for_qid(self, qid: str) -> Dict[str, str]:
        """Return {list_id: answer} for a qid (used during EE rebuild on resume)."""
        return {
            r["list_id"]: r["answer"]
            for r in self.load_jsonl(_ANSWERS_FILE)
            if r["qid"] == qid
        }

    def get_query_summary_qids(self) -> Set[str]:
        """Return set of qids with a saved query summary row."""
        return {r["qid"] for r in self.load_jsonl(_QUERY_SUMMARY_FILE)}

    def load_query_summaries(self) -> List[Dict]:
        """Return all persisted per-query summary rows."""
        return self.load_jsonl(_QUERY_SUMMARY_FILE)

    def load_progress_reports(self) -> List[Dict]:
        """Return all persisted progress snapshots."""
        return self.load_jsonl(_PROGRESS_FILE)

    def build_query_summary_for_qid(self, qid: str) -> Optional[Dict[str, Any]]:
        """Aggregate persisted artifacts into a single per-query summary row."""
        per_list = [r for r in self.load_jsonl(_PER_LIST_FILE) if r["qid"] == qid]
        if not per_list:
            return None

        ee_rec = next((r for r in self.load_jsonl(_EE_FILE) if r["qid"] == qid), {})
        eu_scores = [r["eu_score"] for r in per_list if r.get("eu_score") is not None]
        ild_scores = [r["ild_jaccard"] for r in per_list if r.get("ild_jaccard") is not None]
        jac_scores = [r["jaccard_mean"] for r in per_list if r.get("jaccard_mean") is not None]
        metric_name = per_list[0].get("metric_name")

        return {
            "qid": qid,
            "metric_name": metric_name,
            "expected_utility": sum(eu_scores) / len(eu_scores) if eu_scores else None,
            "max_utility": max(eu_scores) if eu_scores else None,
            "min_utility": min(eu_scores) if eu_scores else None,
            "n_lists": len(per_list),
            "avg_ild_jaccard": sum(ild_scores) / len(ild_scores) if ild_scores else None,
            "avg_jaccard_mean": sum(jac_scores) / len(jac_scores) if jac_scores else None,
            "ee_disparity": ee_rec.get("ee_disparity"),
            "ee_relevance": ee_rec.get("ee_relevance"),
            "ee_difference": ee_rec.get("ee_difference"),
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def build_summary(self) -> Dict:
        """
        Aggregate per-list metrics into per-query summaries.

        Returns a dict keyed by qid with averaged EU and diversity metrics,
        plus EE values.
        """
        query_rows = self.load_query_summaries()
        if not query_rows:
            qids = sorted({r["qid"] for r in self.load_jsonl(_PER_LIST_FILE)})
            query_rows = [
                row for row in (self.build_query_summary_for_qid(qid) for qid in qids)
                if row is not None
            ]
        return {row["qid"]: {k: v for k, v in row.items() if k != "qid"} for row in query_rows}

    def write_summary(self) -> str:
        """Build and write summary.json; returns path."""
        summary = self.build_summary()
        fp = self._path("summary.json")
        with open(fp, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        return fp

    def build_macro_summary(self) -> Dict[str, Any]:
        """Aggregate persisted query summaries into one macro-level metrics row."""
        query_rows = self.load_query_summaries()
        metric_keys = [
            "expected_utility",
            "max_utility",
            "min_utility",
            "avg_ild_jaccard",
            "avg_jaccard_mean",
            "ee_disparity",
            "ee_relevance",
            "ee_difference",
        ]
        macro: Dict[str, Any] = {"n_queries": len(query_rows)}
        if query_rows:
            macro["metric_name"] = query_rows[0].get("metric_name")
        for key in metric_keys:
            values = [row[key] for row in query_rows if row.get(key) is not None]
            macro[key] = sum(values) / len(values) if values else None
        return macro

    def write_macro_summary(self) -> str:
        """Write run-level macro summary to macro_summary.json; returns path."""
        macro = self.build_macro_summary()
        fp = self._path("macro_summary.json")
        with open(fp, "w", encoding="utf-8") as fh:
            json.dump(macro, fh, indent=2, ensure_ascii=False)
        return fp


# ---------------------------------------------------------------------------
# Run-dir factory
# ---------------------------------------------------------------------------

def make_run_dir(run_id: str, base_dir: Optional[str] = None) -> str:
    if base_dir is None:
        base_dir = os.path.join(ROOT, "experiment_runs")
    return os.path.join(base_dir, run_id)
