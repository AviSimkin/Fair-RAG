"""Sequential batch execution for multiple Fair-RAG run settings."""

from __future__ import annotations

import datetime
import json
import os
import sys
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)

from framework.config import RunConfig, setting_id
from framework.runner import ExperimentRunner
from framework.cross_run_analysis import build_macro_comparison_rows


class BatchExperimentRunner:
    """Run multiple experiment configurations sequentially and persist a batch summary."""

    def __init__(self, configs: List[RunConfig], batch_id: Optional[str] = None) -> None:
        self.configs = configs
        self.batch_id = batch_id or datetime.datetime.now().strftime("batch_%Y%m%d_%H%M%S")

    def run_all(self) -> Dict[str, object]:
        run_dirs: List[str] = []
        comparisons: List[Dict[str, object]] = []

        for index, cfg in enumerate(self.configs, start=1):
            print(
                f"[Batch] Starting run {index}/{len(self.configs)}: {setting_id(cfg)}",
                flush=True,
            )
            store = ExperimentRunner(cfg).run()
            run_dirs.append(store.run_dir)

        comparisons = build_macro_comparison_rows(run_dirs)
        batch_summary = {
            "batch_id": self.batch_id,
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "run_dirs": run_dirs,
            "runs": comparisons,
        }
        summary_fp = self._write_batch_summary(batch_summary)
        print(f"[Batch] Summary written to {summary_fp}")
        return batch_summary

    def _write_batch_summary(self, batch_summary: Dict[str, object]) -> str:
        batch_dir = os.path.join(ROOT, "experiment_runs", "batches", self.batch_id)
        os.makedirs(batch_dir, exist_ok=True)
        fp = os.path.join(batch_dir, "batch_summary.json")
        with open(fp, "w", encoding="utf-8") as fh:
            json.dump(batch_summary, fh, indent=2, ensure_ascii=False)
        return fp
