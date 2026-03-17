"""
Post-process normalization for Fair-RAG metrics.

This script normalizes EU (Expected Utility) only, after all required experiments are completed.
EE-D, EE-R, and EE-L are already normalized during experiment runs using theoretical bounds
from the expected_exposure library and should NOT be re-normalized here.

Required inputs for a given (generator, lamp, retriever):
- experiment_results/<generator>/lamp<lamp>/<retriever>/alpha_<a>.json for all alphas
- experiment_results/<generator>/lamp<lamp>/gold/alpha_8.json

Outputs:
- experiment_results/<generator>/lamp<lamp>/<retriever>/alpha_<a>_normalized.json
"""

import os
import argparse
import numpy as np
import json
import copy
import logging
from datetime import datetime


CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
logger = logging.getLogger()


def lamp_utility_metric(lamp_num: int) -> str:
    if lamp_num in {1, 2}:
        return "acc"
    if lamp_num in {3}:
        return "mae"
    return "rouge-l"


def convert_to_higher_the_better(value: float, upper_bound: float) -> float:
    return upper_bound - value


def normalize_minmax(value: float, lower: float, upper: float, default: float = 1.0) -> float:
    if upper == lower:
        return default
    return (value - lower) / (upper - lower)


def load_json(fp: str) -> dict:
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_alphas(alpha_str: str) -> list[int]:
    alphas = []
    for x in alpha_str.split(","):
        x = x.strip()
        if not x:
            continue
        alphas.append(int(x))
    if not alphas:
        raise ValueError("No valid alpha values were provided.")
    return sorted(set(alphas))


def main(args):
    lamp_num: int = args.lamp_num
    generator_name: str = args.generator_name
    retriever_name: str = args.retriever_name
    run_id: str = args.run_id
    all_alphas: list[int] = parse_alphas(args.alphas)
    target_alphas: list[int] = [args.alpha] if args.alpha is not None else all_alphas

    if args.alpha is not None and args.alpha not in all_alphas:
        raise ValueError(f"Requested --alpha {args.alpha} is not included in --alphas {all_alphas}")

    base_model_dir = os.path.join(
        CUR_DIR_PATH,
        "experiment_results",
        generator_name,
        f"lamp{lamp_num}",
        retriever_name,
    )

    gold_results_fp = os.path.join(
        CUR_DIR_PATH,
        "experiment_results",
        generator_name,
        f"lamp{lamp_num}",
        "gold",
        "alpha_8.json",
    )

    model_results_fps: dict[int, str] = {
        alpha: os.path.join(base_model_dir, f"alpha_{alpha}.json") for alpha in all_alphas
    }

    missing_files = []
    if not os.path.exists(gold_results_fp):
        missing_files.append(gold_results_fp)
    for _, fp in model_results_fps.items():
        if not os.path.exists(fp):
            missing_files.append(fp)

    if missing_files:
        missing_msg = "\n".join(missing_files)
        raise FileNotFoundError(
            "Cannot normalize metrics yet. Missing required experiment outputs:\n"
            f"{missing_msg}\n"
            "Run all deterministic-alpha experiments and the gold run first."
        )

    # Build log file name with run_id to avoid overwriting.
    if run_id:
        log_suffix = f"normalize_all_{run_id}.log"
    else:
        log_suffix = "normalize_all.log"

    log_file = os.path.join(base_model_dir, log_suffix)

    logs_dir = os.path.join(
        CUR_DIR_PATH, "logs", generator_name, f"lamp{lamp_num}", retriever_name
    )
    os.makedirs(logs_dir, exist_ok=True)
    tracked_log_file = os.path.join(logs_dir, log_suffix)

    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    for log_path in [log_file, tracked_log_file]:
        fh = logging.FileHandler(log_path, mode="w")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.info(f"{'='*60}")
    logger.info("Post-run Metric Normalization Configuration")
    logger.info(f"{'='*60}")
    logger.info(f"LaMP Number: {lamp_num}")
    logger.info(f"Generator: {generator_name}")
    logger.info(f"Retriever: {retriever_name}")
    logger.info(f"All Alphas (required inputs): {all_alphas}")
    logger.info(f"Target Alphas (outputs): {target_alphas}")
    logger.info(f"{'='*60}\n")

    utility_metric = lamp_utility_metric(lamp_num)

    gold_results_dict = load_json(gold_results_fp)
    model_results_by_alpha: dict[int, dict] = {
        alpha: load_json(fp) for alpha, fp in model_results_fps.items()
    }

    gold_qids = set(gold_results_dict.keys())
    for alpha, results in model_results_by_alpha.items():
        qids = set(results.keys())
        if qids != gold_qids:
            raise ValueError(
                f"QID mismatch for alpha={alpha}. Expected {len(gold_qids)} qids "
                f"(matching gold), got {len(qids)}."
            )

    # Prepare per-alpha copies to write normalized outputs.
    save_dicts: dict[int, dict] = {
        alpha: copy.deepcopy(model_results_by_alpha[alpha]) for alpha in target_alphas
    }

    for qid in gold_results_dict:
        # EE metrics are already normalized by expected_exposure library using theoretical bounds.
        # We only need to normalize EU here.
        
        # EU denominator uses oracle + best model utility across all alphas.
        if lamp_num != 3:
            gold_max_utility = gold_results_dict[qid]["max-utility"]
            model_max_utility = max(
                model_results_by_alpha[a][qid]["max-utility"] for a in all_alphas
            )
        else:
            gold_max_utility = convert_to_higher_the_better(
                gold_results_dict[qid]["min-utility"], upper_bound=4
            )
            model_min_error = min(
                model_results_by_alpha[a][qid]["min-utility"] for a in all_alphas
            )
            model_max_utility = convert_to_higher_the_better(model_min_error, upper_bound=4)

        max_utility = max(gold_max_utility, model_max_utility)

        for alpha in target_alphas:
            model_entry = save_dicts[alpha][qid]

            # EE metrics are already normalized - just copy them
            model_entry["EE"]["disparity"] = model_results_by_alpha[alpha][qid]["EE"]["disparity"]
            model_entry["EE"]["relevance"] = model_results_by_alpha[alpha][qid]["EE"]["relevance"]
            model_entry["EE"]["difference"] = model_results_by_alpha[alpha][qid]["EE"]["difference"]

            # Normalize EU
            model_eu = model_results_by_alpha[alpha][qid]["EU"][utility_metric]
            if lamp_num == 3:
                model_eu = convert_to_higher_the_better(model_eu, upper_bound=4)

            if max_utility == 0:
                normalized_eu = 1.0
            else:
                normalized_eu = model_eu / max_utility

            model_entry["EU"][utility_metric] = normalized_eu

    for alpha in target_alphas:
        save_fp = os.path.join(base_model_dir, f"alpha_{alpha}_normalized.json")
        with open(save_fp, "w", encoding="utf-8") as f:
            json.dump(save_dicts[alpha], f, indent=2)

        normalized_ee_d = [save_dicts[alpha][qid]["EE"]["disparity"] for qid in save_dicts[alpha]]
        normalized_ee_r = [save_dicts[alpha][qid]["EE"]["relevance"] for qid in save_dicts[alpha]]
        normalized_eu = [save_dicts[alpha][qid]["EU"][utility_metric] for qid in save_dicts[alpha]]

        logger.info(f"Alpha={alpha} | queries={len(normalized_eu)}")
        logger.info(
            f"  EE-D (already normalized) mean/std: {float(np.mean(normalized_ee_d)):.6f} / {float(np.std(normalized_ee_d)):.6f}"
        )
        logger.info(
            f"  EE-R (already normalized) mean/std: {float(np.mean(normalized_ee_r)):.6f} / {float(np.std(normalized_ee_r)):.6f}"
        )
        logger.info(
            f"  EU mean/std:   {float(np.mean(normalized_eu)):.6f} / {float(np.std(normalized_eu)):.6f}"
        )
        logger.info(f"  Saved: {save_fp}")

    logger.info(f"\nTracked log saved to: {tracked_log_file} (committed to git)")
    logger.info(f"Normalization completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for handler in logger.handlers:
        handler.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lamp_num",
        type=int,
        required=True,
        help="LaMP number",
    )
    parser.add_argument(
        "--generator_name",
        type=str,
        default="flanT5Base",
        help="Generator model nickname of HF model",
    )
    parser.add_argument(
        "--retriever_name",
        type=str,
        required=True,
        help="Deterministic retriever model nickname. bm25; contriever; splade",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=None,
        help="Optional single alpha to write output for; if omitted, writes all --alphas.",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="1,2,4,8",
        help="Comma-separated alpha values required for normalization bounds, e.g. '1,2,4,8'.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="",
        help="Run ID for timestamped logging (to avoid overwriting)",
    )
    args = parser.parse_args()

    main(args)
