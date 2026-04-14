"""Utilities for modular Fair-RAG experiment execution from notebooks.

This module keeps experiment orchestration reusable across environments:
- weak machine: small query/sample budget
- stronger machine: larger budget or full runs
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import json
import shutil
import subprocess
import sys
from typing import Iterable
import math

import numpy as np
import pandas as pd


@dataclass
class ExperimentConfig:
    root: Path
    python_exe: Path
    generator_name: str = "flanT5Small"
    lamp_num: int = 4
    retriever_name: str = "bm25"
    alphas: tuple[int, ...] = (1, 2, 4, 8)
    max_queries: int | None = 5
    n_samples: int = 5
    k: int = 5
    remove_temp_files: bool = True
    skip_existing: bool = True
    mmr_base_retriever: str = "bm25"
    mmr_lambda: float = 0.7
    run_tag: str = ""  # e.g. "balanced", "weak" — used in run-log folder names
    print_interval: int | None = None  # stdout print every N queries; None = same as checkpoint_interval
    save_llm_outputs: bool = False


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root by walking upward from the given start path."""
    current = (start or Path.cwd()).resolve()
    markers = ("experiment.py", "notebook_experiment_utils.py")

    for candidate in (current, *current.parents):
        if all((candidate / marker).exists() for marker in markers):
            return candidate

    raise FileNotFoundError(
        "Could not locate repository root from "
        f"{current}. Expected markers: {', '.join(markers)}"
    )


def resolve_python_executable(root: Path) -> Path:
    """Resolve a Python executable path that works across Linux/macOS/Windows."""
    candidates: list[Path] = []

    # Prefer project-local virtual environments first for reproducibility.
    candidates.extend(
        [
            root / ".venv" / "bin" / "python",
            root / ".venv" / "Scripts" / "python.exe",
        ]
    )

    # Fall back to the active notebook/kernel interpreter.
    candidates.append(Path(sys.executable).resolve())

    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path:
        venv_root = Path(venv_path)
        candidates.extend(
            [
                venv_root / "bin" / "python",
                venv_root / "Scripts" / "python.exe",
            ]
        )

    for cmd in ("python3", "python"):
        resolved = shutil.which(cmd)
        if resolved:
            candidates.append(Path(resolved))

    seen: set[Path] = set()
    ordered_unique: list[Path] = []
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered_unique.append(candidate)

    for candidate in ordered_unique:
        if candidate.exists():
            return candidate

    looked_in = "\n".join(str(path) for path in ordered_unique)
    raise FileNotFoundError(
        "Could not resolve a Python executable. Checked:\n" + looked_in
    )


def _run_cmd(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    print("\n> " + " ".join(args))
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    proc = subprocess.Popen(
        args,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        output_lines.append(line)
        print(line, end="")

    return_code = proc.wait()
    combined_output = "".join(output_lines)
    if return_code != 0:
        tail = combined_output[-3000:]
        if tail:
            print("--- command tail ---")
            print(tail)
        raise RuntimeError(f"Command failed with exit code {return_code}")

    return subprocess.CompletedProcess(args=args, returncode=return_code, stdout=combined_output)


def _retrieval_results_path(cfg: ExperimentConfig, retriever_name: str) -> Path:
    return (
        cfg.root
        / "retrieval"
        / "retrieval_results"
        / cfg.generator_name
        / retriever_name
        / f"{cfg.lamp_num}.json"
    )


def _ensure_retrieval_results(cfg: ExperimentConfig, retriever_name: str) -> None:
    """Ensure deterministic retrieval artifacts exist before running experiment.py."""
    fp = _retrieval_results_path(cfg, retriever_name)
    if fp.exists():
        return

    print(f"[prep] missing retrieval file: {fp}")

    if retriever_name == "bm25":
        args = [
            str(cfg.python_exe),
            "retrieval/rank_profiles.py",
            "--lamp_num",
            str(cfg.lamp_num),
            "--ranker",
            "bm25",
            "--generator_name",
            cfg.generator_name,
        ]
        _run_cmd(args, cfg.root)
    elif retriever_name == "gold":
        args = [
            str(cfg.python_exe),
            "retrieval/gold_retriever.py",
            "--lamp_num",
            str(cfg.lamp_num),
            "--generator_name",
            cfg.generator_name,
        ]
        _run_cmd(args, cfg.root)
    else:
        raise FileNotFoundError(
            f"Missing retrieval file {fp}. "
            "Auto-precompute is supported for retrievers: bm25, gold."
        )

    if not fp.exists():
        raise FileNotFoundError(f"Retrieval precompute completed but file is still missing: {fp}")


def find_degenerate_score_qids(
    cfg: ExperimentConfig,
    retriever_name: str | None = None,
    split_type: str = "user",
    max_queries: int | None = None,
) -> list[str]:
    """Return qids whose retriever scores collapse to a degenerate min-max denominator.

    These are qids that previously could emit RuntimeWarning during score normalization.
    """
    name = retriever_name or cfg.retriever_name
    source_retriever = cfg.mmr_base_retriever if name == "mmr" else name

    retrieval_fp = _retrieval_results_path(cfg, source_retriever)
    if not retrieval_fp.exists():
        _ensure_retrieval_results(cfg, source_retriever)

    with retrieval_fp.open("r", encoding="utf-8") as f:
        retrieval_results: dict = json.load(f)

    lamp_dir_candidates = [
        cfg.root / "data" / f"lamp_utility_labels_{cfg.generator_name}",
        cfg.root / "data" / "lamp_utility_labels",
    ]
    lamp_dir = next((d for d in lamp_dir_candidates if d.exists()), None)
    if lamp_dir is None:
        raise FileNotFoundError(
            "Could not find a LaMP data directory under data/lamp_utility_labels*"
        )

    inputs_fp = lamp_dir / f"{cfg.lamp_num}_{split_type}_dev_inputs.json"
    if not inputs_fp.exists():
        raise FileNotFoundError(f"Input file not found: {inputs_fp}")

    with inputs_fp.open("r", encoding="utf-8") as f:
        inputs_data = json.load(f)

    ordered_qids = [str(x["id"]) for x in inputs_data]
    limit = cfg.max_queries if max_queries is None else max_queries
    if limit is not None:
        ordered_qids = ordered_qids[:limit]

    bad_qids: list[str] = []
    for qid in ordered_qids:
        rows = retrieval_results.get(qid, [])
        if not rows:
            continue

        scores = np.array([float(pid_score[1]) for pid_score in rows], dtype=float)
        if scores.size == 0:
            continue

        min_value = float(scores.min())
        if min_value < 0:
            scores = scores - min_value

        denom = float(scores.max() - scores.min())
        if (not np.isfinite(denom)) or denom <= 0:
            bad_qids.append(qid)

    return bad_qids


def find_invalid_result_qids(
    cfg: ExperimentConfig,
    retriever_name: str | None = None,
    alphas: Iterable[int] | None = None,
    output_suffix: str = "",
) -> list[str]:
    """Return qids that already have invalid EE/EU values in raw result files."""
    name = retriever_name or cfg.retriever_name
    alpha_values = tuple(alphas) if alphas is not None else tuple(cfg.alphas)

    bad_qids: set[str] = set()
    for alpha in alpha_values:
        fp = raw_results_path(cfg, name, int(alpha), output_suffix=output_suffix)
        if not fp.exists():
            continue
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            continue

        for qid, payload in data.items():
            if not isinstance(payload, dict):
                bad_qids.add(str(qid))
                continue

            ee = payload.get("EE", {}) or {}
            eu = payload.get("EU", {}) or {}

            ee_vals = [
                ee.get("disparity", None),
                ee.get("relevance", None),
                ee.get("difference", None),
            ]
            eu_vals = list(eu.values()) if isinstance(eu, dict) else []
            vals = ee_vals + eu_vals

            invalid = False
            for v in vals:
                try:
                    fv = float(v)
                except Exception:
                    invalid = True
                    break
                if not math.isfinite(fv):
                    invalid = True
                    break

            if invalid:
                bad_qids.add(str(qid))

    return sorted(bad_qids)


def run_experiment_for_alpha(
    cfg: ExperimentConfig,
    retriever_name: str,
    alpha: int,
    deterministic_ranking: bool = False,
    output_suffix: str = "",
) -> None:
    retrieval_source = cfg.mmr_base_retriever if retriever_name == "mmr" else retriever_name
    _ensure_retrieval_results(cfg, retrieval_source)

    out_fp = raw_results_path(cfg, retriever_name, alpha, output_suffix=output_suffix)
    if cfg.skip_existing and out_fp.exists():
        # Only skip if the existing file covers at least as many queries as requested.
        # A file from a smaller run (e.g. 30-query weak) must NOT block a 833-query balanced run.
        _should_skip = True
        if cfg.max_queries is not None:
            try:
                with out_fp.open() as _f:
                    _existing_count = len(json.load(_f))
                if _existing_count < cfg.max_queries:
                    print(
                        f"[stale] {out_fp.name} has {_existing_count} queries "
                        f"but {cfg.max_queries} requested — rerunning"
                    )
                    _should_skip = False
            except Exception:
                _should_skip = False  # unreadable file → rerun to be safe
        if _should_skip:
            # Load and summarise cached results so the user can see them inline.
            try:
                with out_fp.open() as _f:
                    _cached = json.load(_f)
                _n = len(_cached)
                _ee_d_vals = [v["EE"]["disparity"] for v in _cached.values() if "EE" in v]
                _ee_r_vals = [v["EE"]["relevance"] for v in _cached.values() if "EE" in v]
                _eu_vals   = [next(iter(v["EU"].values())) for v in _cached.values() if "EU" in v]
                _mean = lambda lst: sum(lst) / len(lst) if lst else float("nan")
                print(
                    f"[skip] {out_fp.name}  ({_n} queries) "
                    f"EE-D={_mean(_ee_d_vals):.4f}  "
                    f"EE-R={_mean(_ee_r_vals):.4f}  "
                    f"EU={_mean(_eu_vals):.4f}"
                )
            except Exception:
                print(f"[skip] existing: {out_fp}")
            return

    n_samples = 1 if deterministic_ranking else cfg.n_samples

    args = [
        str(cfg.python_exe),
        "experiment.py",
        "--generator_name",
        cfg.generator_name,
        "--lamp_num",
        str(cfg.lamp_num),
        "--retriever_name",
        retriever_name,
        "--alpha",
        str(alpha),
        "--k",
        str(cfg.k),
        "--n_samples",
        str(n_samples),
    ]
    if deterministic_ranking:
        args.append("--deterministic_ranking")
    if retriever_name == "mmr":
        args.extend(["--mmr_base_retriever", cfg.mmr_base_retriever])
        args.extend(["--mmr_lambda", str(cfg.mmr_lambda)])
    if output_suffix:
        args.extend(["--output_suffix", output_suffix])
    if cfg.max_queries is not None:
        args.extend(["--max_queries", str(cfg.max_queries)])
    if cfg.remove_temp_files:
        args.append("--remove_temp_files")
    if cfg.run_tag:
        args.extend(["--run_tag", cfg.run_tag])
    if cfg.print_interval is not None:
        args.extend(["--print_interval", str(cfg.print_interval)])
    if cfg.save_llm_outputs:
        args.append("--save_llm_outputs")
    _run_cmd(args, cfg.root)


def run_targeted_qids_for_alpha(
    cfg: ExperimentConfig,
    retriever_name: str,
    alpha: int,
    qids: Iterable[str],
    deterministic_ranking: bool = False,
    output_suffix: str = "",
    recompute_target_qids: bool = True,
) -> None:
    """Recompute only selected qids and merge into existing alpha output."""
    qid_list = [str(q).strip() for q in qids if str(q).strip()]
    if not qid_list:
        raise ValueError("run_targeted_qids_for_alpha requires at least one qid")

    retrieval_source = cfg.mmr_base_retriever if retriever_name == "mmr" else retriever_name
    _ensure_retrieval_results(cfg, retrieval_source)

    qids_dir = cfg.root / "experiment_results" / "runs" / "target_qids"
    qids_dir.mkdir(parents=True, exist_ok=True)
    qids_fp = qids_dir / (
        f"{retriever_name}_alpha{alpha}"
        f"{output_suffix if output_suffix.startswith('_') or not output_suffix else '_' + output_suffix}"
        "_qids.json"
    )
    with qids_fp.open("w", encoding="utf-8") as f:
        json.dump({"qids": sorted(set(qid_list))}, f, indent=2)

    n_samples = 1 if deterministic_ranking else cfg.n_samples

    args = [
        str(cfg.python_exe),
        "experiment.py",
        "--generator_name",
        cfg.generator_name,
        "--lamp_num",
        str(cfg.lamp_num),
        "--retriever_name",
        retriever_name,
        "--alpha",
        str(alpha),
        "--k",
        str(cfg.k),
        "--n_samples",
        str(n_samples),
        "--only_qids_file",
        str(qids_fp),
    ]
    if recompute_target_qids:
        args.append("--recompute_target_qids")
    if deterministic_ranking:
        args.append("--deterministic_ranking")
    if retriever_name == "mmr":
        args.extend(["--mmr_base_retriever", cfg.mmr_base_retriever])
        args.extend(["--mmr_lambda", str(cfg.mmr_lambda)])
    if output_suffix:
        args.extend(["--output_suffix", output_suffix])
    if cfg.max_queries is not None:
        args.extend(["--max_queries", str(cfg.max_queries)])
    if cfg.remove_temp_files:
        args.append("--remove_temp_files")
    if cfg.run_tag:
        args.extend(["--run_tag", cfg.run_tag])
    if cfg.print_interval is not None:
        args.extend(["--print_interval", str(cfg.print_interval)])
    if cfg.save_llm_outputs:
        args.append("--save_llm_outputs")

    print(f"[targeted] recomputing {len(set(qid_list))} qids via {qids_fp}")
    _run_cmd(args, cfg.root)


def run_gold_baseline(cfg: ExperimentConfig, alpha: int = 8) -> None:
    run_experiment_for_alpha(cfg, retriever_name="gold", alpha=alpha)


def run_deterministic_reference(
    cfg: ExperimentConfig,
    alpha: int = 1,
    output_suffix: str = "_deterministic",
) -> None:
    """Run a deterministic ranking baseline using the configured retriever."""
    run_experiment_for_alpha(
        cfg,
        retriever_name=cfg.retriever_name,
        alpha=alpha,
        deterministic_ranking=True,
        output_suffix=output_suffix,
    )


def run_mmr_deterministic(
    cfg: ExperimentConfig,
    alpha: int = 1,
    output_suffix: str = "_mmr_deterministic",
) -> None:
    """Run one deterministic MMR retrieval baseline (no PL sampling)."""
    run_experiment_for_alpha(
        cfg,
        retriever_name="mmr",
        alpha=alpha,
        deterministic_ranking=True,
        output_suffix=output_suffix,
    )


def run_gold_deterministic_reference(
    cfg: ExperimentConfig,
    alpha: int = 1,
    output_suffix: str = "_gold_deterministic",
) -> None:
    """Run deterministic gold ranking (no PL sampling) — oracle fairness sanity check."""
    run_experiment_for_alpha(
        cfg,
        retriever_name="gold",
        alpha=alpha,
        deterministic_ranking=True,
        output_suffix=output_suffix,
    )


def run_retriever_grid(cfg: ExperimentConfig) -> None:
    for alpha in cfg.alphas:
        run_experiment_for_alpha(cfg, retriever_name=cfg.retriever_name, alpha=alpha)


def normalize_eu_grid(cfg: ExperimentConfig, normalization_scope: str = "all-settings") -> None:
    for alpha in cfg.alphas:
        out_fp = normalized_results_path(cfg, alpha)
        if cfg.skip_existing and out_fp.exists():
            print(f"[skip] existing: {out_fp}")
            continue

        args = [
            str(cfg.python_exe),
            "normalize_eu.py",
            "--generator_name",
            cfg.generator_name,
            "--lamp_num",
            str(cfg.lamp_num),
            "--retriever_name",
            cfg.retriever_name,
            "--alpha",
            str(alpha),
            "--normalization_scope",
            normalization_scope,
        ]
        _run_cmd(args, cfg.root)


def normalize_eu_for_retriever(
    cfg: ExperimentConfig,
    retriever_name: str,
    alpha: int,
    output_suffix: str = "",
    normalization_scope: str = "all-settings",
) -> None:
    """Normalize EU for a specific retriever and alpha.

    normalization_scope:
    - legacy: current retriever (requested alphas) plus gold alpha_8
    - all-settings: any raw result file for the same generator/lamp
    """
    base_path = (
        cfg.root
        / "experiment_results"
        / cfg.generator_name
        / f"lamp{cfg.lamp_num}"
        / retriever_name
    )
    
    suffix = output_suffix
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    
    # Expected input file (raw results)
    raw_fp = base_path / f"alpha_{alpha}{suffix}.json"
    if not raw_fp.exists():
        print(f"[skip] raw results not found: {raw_fp}")
        return
    
    # Expected output file (normalized)
    out_fp = base_path / f"alpha_{alpha}{suffix}_normalized.json"
    if cfg.skip_existing and out_fp.exists():
        print(f"[skip] existing normalized: {out_fp}")
        return
    
    args = [
        str(cfg.python_exe),
        "normalize_eu.py",
        "--generator_name",
        cfg.generator_name,
        "--lamp_num",
        str(cfg.lamp_num),
        "--retriever_name",
        retriever_name,
        "--alpha",
        str(alpha),
        "--all_alphas",
        str(alpha),
        "--normalization_scope",
        normalization_scope,
    ]
    if suffix:
        args.extend(["--input_suffix", suffix])
    _run_cmd(args, cfg.root)


def raw_results_path(
    cfg: ExperimentConfig,
    retriever_name: str,
    alpha: int,
    output_suffix: str = "",
) -> Path:
    suffix = output_suffix
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"

    return (
        cfg.root
        / "experiment_results"
        / cfg.generator_name
        / f"lamp{cfg.lamp_num}"
        / retriever_name
        / f"alpha_{alpha}{suffix}.json"
    )


def reset_run_outputs(cfg: ExperimentConfig, include_gold: bool = True) -> None:
    """Delete raw/normalized outputs for configured alphas so reruns are clean."""
    deleted = 0
    for alpha in cfg.alphas:
        raw_fp = raw_results_path(cfg, cfg.retriever_name, alpha)
        norm_fp = normalized_results_path(cfg, alpha)
        if raw_fp.exists():
            raw_fp.unlink()
            deleted += 1
        if norm_fp.exists():
            norm_fp.unlink()
            deleted += 1

    if include_gold:
        gold_fp = raw_results_path(cfg, "gold", 8)
        if gold_fp.exists():
            gold_fp.unlink()
            deleted += 1

    mmr_det_fp = raw_results_path(cfg, "mmr", 1, output_suffix="_mmr_deterministic")
    if mmr_det_fp.exists():
        mmr_det_fp.unlink()
        deleted += 1

    mmr_det_norm_fp = (
        cfg.root
        / "experiment_results"
        / cfg.generator_name
        / f"lamp{cfg.lamp_num}"
        / "mmr"
        / "alpha_1_mmr_deterministic_normalized.json"
    )
    if mmr_det_norm_fp.exists():
        mmr_det_norm_fp.unlink()
        deleted += 1

    print(f"reset_run_outputs: deleted {deleted} file(s)")


def assert_consistent_qids_across_alphas(cfg: ExperimentConfig, retriever_name: str) -> None:
    """Ensure every alpha result file has the same qid set before normalization."""
    qid_sets: dict[int, set[str]] = {}
    missing_files: list[str] = []

    for alpha in cfg.alphas:
        fp = raw_results_path(cfg, retriever_name, alpha)
        if not fp.exists():
            missing_files.append(str(fp))
            continue
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        qid_sets[alpha] = set(data.keys())

    if missing_files:
        raise FileNotFoundError(
            "Missing alpha result files before normalization:\n" + "\n".join(missing_files)
        )

    baseline_alpha = cfg.alphas[0]
    baseline = qid_sets[baseline_alpha]
    for alpha, qids in qid_sets.items():
        if qids != baseline:
            missing_vs_base = sorted(list(baseline - qids))
            extra_vs_base = sorted(list(qids - baseline))
            raise ValueError(
                "Inconsistent qid sets across alpha files. "
                f"baseline alpha={baseline_alpha} has {len(baseline)} qids, "
                f"alpha={alpha} has {len(qids)} qids. "
                f"missing_vs_baseline={missing_vs_base[:5]}, "
                f"extra_vs_baseline={extra_vs_base[:5]}. "
                "Set FORCE_FRESH_RUN=True and rerun the experiment stage."
            )


def normalized_results_path(cfg: ExperimentConfig, alpha: int) -> Path:
    return (
        cfg.root
        / "experiment_results"
        / cfg.generator_name
        / f"lamp{cfg.lamp_num}"
        / cfg.retriever_name
        / f"alpha_{alpha}_normalized.json"
    )


def load_normalized_rows(cfg: ExperimentConfig) -> pd.DataFrame:
    if cfg.retriever_name == "gold":
        raise ValueError(
            "gold is the oracle reference run and must not be used for published statistics. "
            "Set cfg.retriever_name to a non-oracle retriever such as bm25."
        )

    rows: list[dict] = []
    for alpha in cfg.alphas:
        fp = normalized_results_path(cfg, alpha)
        if not fp.exists():
            raise FileNotFoundError(f"Missing normalized file: {fp}")
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for qid, entry in data.items():
            rows.append(
                {
                    "retriever": cfg.retriever_name,
                    "alpha": alpha,
                    "qid": qid,
                    "ee_d": entry["EE"]["disparity"],
                    "ee_r": entry["EE"]["relevance"],
                    "eu": entry["EU"]["rouge-l"],
                }
            )
    return pd.DataFrame(rows)


def load_raw_rows(
    cfg: ExperimentConfig,
    retriever_name: str,
    alphas: tuple[int, ...],
    output_suffix: str = "",
) -> pd.DataFrame:
    rows: list[dict] = []
    for alpha in alphas:
        fp = raw_results_path(cfg, retriever_name, alpha, output_suffix=output_suffix)
        if not fp.exists():
            raise FileNotFoundError(f"Missing raw file: {fp}")
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for qid, entry in data.items():
            eu_dict = entry.get("EU", {})
            eu_value = float(next(iter(eu_dict.values()))) if eu_dict else float("nan")
            rows.append(
                {
                    "retriever": retriever_name,
                    "alpha": alpha,
                    "qid": qid,
                    "ee_d": float(entry["EE"]["disparity"]),
                    "ee_r": float(entry["EE"]["relevance"]),
                    "eu": eu_value,
                }
            )
    return pd.DataFrame(rows)


def load_normalized_rows_for_retriever(
    cfg: ExperimentConfig,
    retriever_name: str,
    alphas: tuple[int, ...],
    output_suffix: str = "",
) -> pd.DataFrame:
    """Load normalized results for any retriever (e.g., MMR deterministic)."""
    rows: list[dict] = []
    for alpha in alphas:
        suffix = output_suffix
        if suffix and not suffix.startswith("_"):
            suffix = f"_{suffix}"
        
        fp = (
            cfg.root
            / "experiment_results"
            / cfg.generator_name
            / f"lamp{cfg.lamp_num}"
            / retriever_name
            / f"alpha_{alpha}{suffix}_normalized.json"
        )
        
        if not fp.exists():
            raise FileNotFoundError(
                f"Missing normalized file: {fp}. "
                f"Run normalize_eu_for_retriever(cfg, '{retriever_name}', {alpha}, '{output_suffix}') first."
            )
        
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        for qid, entry in data.items():
            rows.append(
                {
                    "retriever": retriever_name,
                    "alpha": alpha,
                    "qid": qid,
                    "ee_d": entry["EE"]["disparity"],
                    "ee_r": entry["EE"]["relevance"],
                    "eu": entry["EU"]["rouge-l"],
                }
            )
    return pd.DataFrame(rows)


def add_ee_d_interval_bins(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.000001]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    out = df.copy()
    out["ee_d_interval"] = pd.cut(
        out["ee_d"], bins=bins, labels=labels, include_lowest=True, right=True
    )
    return out


def summarize_by_interval(df_binned: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all queries (across all alphas) by EE-D interval only."""
    summary = (
        df_binned.groupby("ee_d_interval", dropna=False)
        .agg(
            n_queries=("qid", "nunique"),
            mean_eu=("eu", "mean"),
            mean_ee_d=("ee_d", "mean"),
            mean_ee_r=("ee_r", "mean"),
        )
        .reset_index()
        .sort_values("ee_d_interval")
    )
    return summary


def save_interval_outputs(
    cfg: ExperimentConfig,
    summary: pd.DataFrame,
    output_dir_name: str,
) -> Path:
    out_dir = (
        cfg.root
        / "experiment_results"
        / cfg.generator_name
        / f"lamp{cfg.lamp_num}"
        / cfg.retriever_name
        / output_dir_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_fp = out_dir / "ee_d_interval_summary.csv"
    summary.to_csv(summary_fp, index=False)
    return summary_fp


def save_per_query_log(
    cfg: ExperimentConfig,
    df: pd.DataFrame,
    output_dir_name: str,
) -> Path:
    """Save the full per-query DataFrame (all alphas, all metrics) as CSV for later analysis."""
    out_dir = (
        cfg.root
        / "experiment_results"
        / cfg.generator_name
        / f"lamp{cfg.lamp_num}"
        / cfg.retriever_name
        / output_dir_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / "per_query_all_metrics.csv"
    df.to_csv(fp, index=False)
    return fp


def interval_pvalues_vs_deterministic(
    stochastic_df: pd.DataFrame,
    deterministic_df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Compute paired Wilcoxon p-values per alpha and EE-D interval vs deterministic reference."""
    from scipy.stats import wilcoxon

    if metric not in {"ee_r", "eu"}:
        raise ValueError("metric must be one of: ee_r, eu")

    det_by_qid = deterministic_df.groupby("qid", as_index=False)[metric].mean()
    rows: list[dict] = []

    alphas = sorted([int(x) for x in stochastic_df["alpha"].dropna().unique()])
    intervals = [str(x) for x in stochastic_df["ee_d_interval"].dropna().unique()]
    for alpha in alphas:
        for interval in sorted(intervals):
            st = stochastic_df[
                (stochastic_df["alpha"] == alpha)
                & (stochastic_df["ee_d_interval"].astype(str) == interval)
            ][["qid", metric]].dropna()
            paired = st.merge(det_by_qid, on="qid", how="inner", suffixes=("_st", "_det"))

            n = len(paired)
            if n < 2:
                rows.append(
                    {
                        "alpha": alpha,
                        "ee_d_interval": interval,
                        "metric": metric,
                        "n_qids": n,
                        "mean_diff_st_minus_det": float("nan"),
                        "p_two_sided": float("nan"),
                        "p_st_greater": float("nan"),
                        "p_st_less": float("nan"),
                    }
                )
                continue

            x = paired[f"{metric}_st"].to_numpy()
            y = paired[f"{metric}_det"].to_numpy()
            mean_diff = float((x - y).mean())

            # Wilcoxon is undefined when all paired differences are exactly zero.
            # For that degenerate case, treat the runs as indistinguishable.
            if (x == y).all():
                p_two = 1.0
                p_gt = 1.0
                p_lt = 1.0
            else:
                p_two = float(
                    wilcoxon(x, y, alternative="two-sided", zero_method="wilcox").pvalue
                )
                p_gt = float(
                    wilcoxon(x, y, alternative="greater", zero_method="wilcox").pvalue
                )
                p_lt = float(
                    wilcoxon(x, y, alternative="less", zero_method="wilcox").pvalue
                )

            rows.append(
                {
                    "alpha": alpha,
                    "ee_d_interval": interval,
                    "metric": metric,
                    "n_qids": n,
                    "mean_diff_st_minus_det": mean_diff,
                    "p_two_sided": p_two,
                    "p_st_greater": p_gt,
                    "p_st_less": p_lt,
                }
            )

    return pd.DataFrame(rows).sort_values(["metric", "alpha", "ee_d_interval"])


def save_pvalue_outputs(
    cfg: ExperimentConfig,
    pvalues_df: pd.DataFrame,
    output_dir_name: str,
) -> Path:
    out_dir = (
        cfg.root
        / "experiment_results"
        / cfg.generator_name
        / f"lamp{cfg.lamp_num}"
        / cfg.retriever_name
        / output_dir_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / "pvalues_vs_deterministic_by_interval.csv"
    pvalues_df.to_csv(fp, index=False)
    return fp


def ensure_paths_exist(paths: Iterable[Path]) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(str(x) for x in missing))
