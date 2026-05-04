"""Sequential batch execution for multiple Fair-RAG run settings."""

from __future__ import annotations

import copy
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
from framework.artifacts import RunRegistry


class BatchExperimentRunner:
    """Run multiple experiment configurations sequentially and persist a batch summary."""

    def __init__(
        self,
        configs: List[RunConfig],
        batch_id: Optional[str] = None,
        reuse_policy: str = "smart",
    ) -> None:
        self.configs = configs
        self.batch_id = batch_id or datetime.datetime.now().strftime("batch_%Y%m%d_%H%M%S")
        self.reuse_policy = reuse_policy
        self.registry = RunRegistry()

    def run_all(self) -> Dict[str, object]:
        run_dirs: List[str] = []
        comparisons: List[Dict[str, object]] = []
        decisions: List[Dict[str, object]] = []

        for index, cfg in enumerate(self.configs, start=1):
            cfg_to_run, decision = self._resolve_config_action(cfg)
            sid = decision["setting_id"]

            if decision["action"] == "skip_completed":
                print(
                    f"[Batch] Skipping run {index}/{len(self.configs)}: {sid} "
                    f"(reusing completed {decision['run_id']})",
                    flush=True,
                )
                run_dirs.append(decision["run_dir"])
                decisions.append(decision)
                self._write_batch_summary(
                    self._build_batch_summary(run_dirs=run_dirs, decisions=decisions)
                )
                continue

            verb = "Resuming" if decision["action"] == "resume_existing" else "Starting"
            print(
                f"[Batch] {verb} run {index}/{len(self.configs)}: {sid}",
                flush=True,
            )
            store = ExperimentRunner(cfg_to_run).run()
            run_dirs.append(store.run_dir)
            decision["run_dir"] = store.run_dir
            decisions.append(decision)
            self._write_batch_summary(
                self._build_batch_summary(run_dirs=run_dirs, decisions=decisions)
            )

        comparisons = build_macro_comparison_rows(run_dirs)
        batch_summary = self._build_batch_summary(run_dirs=run_dirs, decisions=decisions)
        batch_summary["runs"] = comparisons
        summary_fp = self._write_batch_summary(batch_summary)
        print(f"[Batch] Summary written to {summary_fp}")
        return batch_summary

    def _resolve_config_action(self, cfg: RunConfig) -> tuple[RunConfig, Dict[str, object]]:
        sid = setting_id(cfg)
        decision: Dict[str, object] = {
            "setting_id": sid,
            "action": "start_fresh",
            "run_id": cfg.run_id,
            "run_dir": None,
            "reuse_policy": self.reuse_policy,
        }

        if cfg.run_id is not None:
            decision["action"] = "explicit_run_id"
            return cfg, decision

        if self.reuse_policy == "fresh":
            return cfg, decision

        completed = self.registry.find_latest_completed(cfg, setting_id=sid)
        if completed is not None and self.reuse_policy == "smart":
            manifest = completed["manifest"]
            decision.update(
                {
                    "action": "skip_completed",
                    "run_id": manifest.get("run_id"),
                    "run_dir": completed["run_dir"],
                }
            )
            return cfg, decision

        resumable = self.registry.find_latest_resumable(cfg, setting_id=sid)
        if resumable is not None:
            manifest = resumable["manifest"]
            resumed_cfg = copy.deepcopy(cfg)
            resumed_cfg.resume = True
            resumed_cfg.run_id = manifest.get("run_id")
            decision.update(
                {
                    "action": "resume_existing",
                    "run_id": manifest.get("run_id"),
                    "run_dir": resumable["run_dir"],
                }
            )
            return resumed_cfg, decision

        return cfg, decision

    def _build_batch_summary(
        self,
        run_dirs: List[str],
        decisions: List[Dict[str, object]],
    ) -> Dict[str, object]:
        return {
            "batch_id": self.batch_id,
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "reuse_policy": self.reuse_policy,
            "run_dirs": run_dirs,
            "decisions": decisions,
        }

    def _write_batch_summary(self, batch_summary: Dict[str, object]) -> str:
        batch_dir = os.path.join(ROOT, "experiment_runs", "batches", self.batch_id)
        os.makedirs(batch_dir, exist_ok=True)
        fp = os.path.join(batch_dir, "batch_summary.json")
        with open(fp, "w", encoding="utf-8") as fh:
            json.dump(batch_summary, fh, indent=2, ensure_ascii=False)
        return fp
