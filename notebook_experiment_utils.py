"""Utilities for modular Fair-RAG experiment execution from notebooks.

This module keeps experiment orchestration reusable across environments:
- weak machine: small query/sample budget
- stronger machine: larger budget or full runs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess
from typing import Iterable

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


def _run_cmd(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    print("\n> " + " ".join(args))
    proc = subprocess.run(args, cwd=cwd, text=False, capture_output=True)
    stdout_text = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
    stderr_text = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
    if stdout_text:
        print(stdout_text[-3000:])
    if proc.returncode != 0:
        if stderr_text:
            print("--- stderr ---")
            print(stderr_text[-3000:])
        raise RuntimeError(f"Command failed with exit code {proc.returncode}")
    return proc


def run_experiment_for_alpha(
    cfg: ExperimentConfig,
    retriever_name: str,
    alpha: int,
    deterministic_ranking: bool = False,
    output_suffix: str = "",
) -> None:
    out_fp = raw_results_path(cfg, retriever_name, alpha, output_suffix=output_suffix)
    if cfg.skip_existing and out_fp.exists():
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


def run_retriever_grid(cfg: ExperimentConfig) -> None:
    for alpha in cfg.alphas:
        run_experiment_for_alpha(cfg, retriever_name=cfg.retriever_name, alpha=alpha)


def normalize_eu_grid(cfg: ExperimentConfig) -> None:
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
        ]
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
