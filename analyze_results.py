"""
analyze_results.py
==================
Reads normalized experiment results and produces:

1. EE-D interval table  — per alpha: mean ± std of normalized EE-D (disparity)
   across all queries, plus the [min, max] range.  This matches the paper figure
   where each alpha setting is shown with a spread of EE-D values.

2. EU difference table  — mean normalized EU per alpha, and the percentage-point
   difference relative to the baseline (alpha=1, least fair).

Outputs a human-readable text report and a CSV file for further plotting.
"""

import argparse
import json
import os
import statistics

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
ALPHAS = [1, 2, 4, 8]


def load_results(generator, lamp_num, retriever, alpha, normalized=True):
    suffix = "_normalized" if normalized else ""
    fp = os.path.join(
        CUR_DIR_PATH,
        "experiment_results",
        generator,
        f"lamp{lamp_num}",
        retriever,
        f"alpha_{alpha}{suffix}.json",
    )
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        return json.load(f)


def compute_stats(values):
    if not values:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan")}
    n = len(values)
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": std, "min": min(values), "max": max(values)}


def main(args):
    generator = args.generator_name
    lamp_num = args.lamp_num
    retriever = args.retriever_name
    metric_name = "rouge-l"  # LaMP 4 uses ROUGE-L

    print(f"\n{'='*65}")
    print(f"  Fair-RAG Experiment Recreation Analysis")
    print(f"  Generator: {generator}  |  LaMP: {lamp_num}  |  Retriever: {retriever}")
    print(f"{'='*65}\n")

    # -------------------------------------------------------------------
    # Collect per-query EE-D and EU for each alpha
    # -------------------------------------------------------------------
    alpha_stats = {}
    baseline_eu = None  # alpha=1 mean EU (for difference computation)

    for alpha in ALPHAS:
        data = load_results(generator, lamp_num, retriever, alpha, normalized=True)
        if data is None:
            print(f"  [WARN] No normalized results for alpha={alpha}; skipping.")
            continue

        ee_d_vals = []
        eu_vals = []
        for qid, entry in data.items():
            ee = entry.get("EE", {})
            eu = entry.get("EU", {})
            if "disparity" in ee:
                ee_d_vals.append(ee["disparity"])
            if metric_name in eu:
                eu_vals.append(eu[metric_name])

        alpha_stats[alpha] = {
            "ee_d": compute_stats(ee_d_vals),
            "eu":   compute_stats(eu_vals),
        }
        if alpha == 1:
            baseline_eu = alpha_stats[alpha]["eu"]["mean"]

    if not alpha_stats:
        print("No normalized result files found. Run run_recreation.ps1 first.")
        return

    # -------------------------------------------------------------------
    # Table 1: EE-D intervals
    # -------------------------------------------------------------------
    col = 12
    print("Table 1 — Normalized EE-D (Disparity)  [lower = fairer]")
    print(f"  {'alpha':>6}  {'n':>5}  {'mean':>8}  {'±std':>8}  {'min':>8}  {'max':>8}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for alpha in sorted(alpha_stats):
        s = alpha_stats[alpha]["ee_d"]
        print(
            f"  {alpha:>6}  {s['n']:>5}  {s['mean']:>8.4f}  "
            f"{s['std']:>8.4f}  {s['min']:>8.4f}  {s['max']:>8.4f}"
        )

    # -------------------------------------------------------------------
    # Table 2: EU and EU difference
    # -------------------------------------------------------------------
    print(f"\nTable 2 — Normalized EU ({metric_name})  "
          f"[baseline = alpha=1, difference = alpha_X − baseline]")
    print(f"  {'alpha':>6}  {'n':>5}  {'mean EU':>9}  {'diff vs α=1':>12}  "
          f"{'min EU':>8}  {'max EU':>8}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*9}  {'-'*12}  {'-'*8}  {'-'*8}")
    for alpha in sorted(alpha_stats):
        s = alpha_stats[alpha]["eu"]
        diff = (s["mean"] - baseline_eu) if baseline_eu is not None else float("nan")
        diff_str = f"{diff:+.4f}" if not (diff != diff) else "  n/a"
        print(
            f"  {alpha:>6}  {s['n']:>5}  {s['mean']:>9.4f}  "
            f"{diff_str:>12}  {s['min']:>8.4f}  {s['max']:>8.4f}"
        )

    # -------------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------------
    csv_path = os.path.join(
        CUR_DIR_PATH,
        "experiment_results",
        generator,
        f"lamp{lamp_num}",
        retriever,
        "analysis_summary.csv",
    )
    with open(csv_path, "w") as f:
        f.write("alpha,n_queries,"
                "ee_d_mean,ee_d_std,ee_d_min,ee_d_max,"
                "eu_mean,eu_std,eu_min,eu_max,eu_diff_vs_alpha1\n")
        for alpha in sorted(alpha_stats):
            s_d = alpha_stats[alpha]["ee_d"]
            s_u = alpha_stats[alpha]["eu"]
            diff = s_u["mean"] - baseline_eu if baseline_eu is not None else float("nan")
            f.write(
                f"{alpha},{s_d['n']},"
                f"{s_d['mean']:.6f},{s_d['std']:.6f},{s_d['min']:.6f},{s_d['max']:.6f},"
                f"{s_u['mean']:.6f},{s_u['std']:.6f},{s_u['min']:.6f},{s_u['max']:.6f},"
                f"{diff:.6f}\n"
            )
    print(f"\n  CSV saved to: {csv_path}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze Fair-RAG experiment results: EE-D intervals + EU differences."
    )
    parser.add_argument("--generator_name", type=str, default="flanT5Small")
    parser.add_argument("--lamp_num", type=int, default=4)
    parser.add_argument("--retriever_name", type=str, default="bm25")
    args = parser.parse_args()
    main(args)
