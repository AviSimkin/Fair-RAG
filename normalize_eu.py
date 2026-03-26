"""
Normalization of EU is done to get the percentage of closeness to the approximated optimal utility.
I.e, the normalized EU value indicates how close the EU is to the max utility.
Per query, max utility can be approximated by \max(max-utility of current model, max-utility of gold+currentGenerator)
Then, the Normalized EU can be obtained by EU divided by its max utility

For 'lower the better' metrics (e.g. MAE), we convert the values by (utility_upper_bound - EU).
This is because, without the conversion, the optimal utility is the minimum utility value which makes the inconsistency in max-normalization.
The conversion changes the value to 'higher the better' metric, allowing us to perform the same normalization operation as above.
The conversion is done the same when getting the max utility.
"""

import os
import argparse
import numpy as np
import json
import copy


CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def lamp_utility_metric(lamp_num) -> str:
    if lamp_num in {1, 2}:
        metric = "acc"
    elif lamp_num in {3}:
        metric = "mae"
    else:
        metric = "rouge-l"
    return metric


def convert_to_higher_the_better(value, upper_bound):
    return upper_bound - value


def parse_alphas(alpha_values: str | None) -> list[int]:
    if not alpha_values:
        return [1, 2, 4, 8]
    return [int(value.strip()) for value in alpha_values.split(",") if value.strip()]


def main(args):
    LAMP_NUM: int = args.lamp_num
    GENERATOR_NAME = args.generator_name
    RETRIEVER_NAME = args.retriever_name  # deterministic retriever
    ALPHA: int = args.alpha  # for current alpha's result to normalize
    ALPHAS: list[int] = parse_alphas(args.all_alphas)
    INPUT_SUFFIX: str = args.input_suffix or ""
    if INPUT_SUFFIX and not INPUT_SUFFIX.startswith("_"):
        INPUT_SUFFIX = f"_{INPUT_SUFFIX}"
    # access to gold model's experiment results
    GOLD_RESULTS_FP = os.path.join(
        CUR_DIR_PATH,
        "experiment_results",
        GENERATOR_NAME,
        f"lamp{LAMP_NUM}",
        "gold",
        # safe to say gold retriever is when alpha is 8 (put all relevant docs above non-relevant)
        "alpha_8.json",
    )
    with open(GOLD_RESULTS_FP, "r") as f:
        gold_results_dict: dict = json.load(f)
    f.close()
    del GOLD_RESULTS_FP
    # access to current model's experiment results
    MODEL_RESULTS_FP = os.path.join(
        CUR_DIR_PATH,
        "experiment_results",
        GENERATOR_NAME,
        f"lamp{LAMP_NUM}",
        RETRIEVER_NAME,
        f"alpha_{ALPHA}{INPUT_SUFFIX}.json",
    )
    with open(MODEL_RESULTS_FP, "r") as f:
        model_results_dict: dict = json.load(f)
    f.close()
    del MODEL_RESULTS_FP

    # access to all alphas experiment results
    ALL_ALPHAS_MODEL_RESULTS_FP: list[str] = []
    for alpha in ALPHAS:
        ALL_ALPHAS_MODEL_RESULTS_FP.append(
            os.path.join(
                CUR_DIR_PATH,
                "experiment_results",
                GENERATOR_NAME,
                f"lamp{LAMP_NUM}",
                RETRIEVER_NAME,
                f"alpha_{alpha}{INPUT_SUFFIX}.json",
            )
        )
    all_alpha_model_results: list[dict] = []
    for fp in ALL_ALPHAS_MODEL_RESULTS_FP:
        with open(fp, "r") as f:
            all_alpha_model_results.append(json.load(f))
        f.close()

    # save path of normalized EU
    SAVE_FP = os.path.join(
        CUR_DIR_PATH,
        "experiment_results",
        GENERATOR_NAME,
        f"lamp{LAMP_NUM}",
        RETRIEVER_NAME,
        f"alpha_{ALPHA}{INPUT_SUFFIX}_normalized.json",
    )

    utility_metric = lamp_utility_metric(LAMP_NUM)
    save_dict = copy.deepcopy(model_results_dict)
    missing_in_gold = 0
    missing_in_some_alpha = 0

    # iterate over qids in the current model run file to avoid key mismatch
    for qid in model_results_dict:
        # Getting max utility
        if not LAMP_NUM == 3:
            model_eu: float = model_results_dict[qid]["EU"][utility_metric]

            # fallback baseline from current alpha's own best sample
            model_max_utility = model_results_dict[qid]["max-utility"]

            if qid in gold_results_dict:
                gold_max_utility = gold_results_dict[qid]["max-utility"]
            else:
                gold_max_utility = model_max_utility
                missing_in_gold += 1

            # get model's max utility across all alphas
            for single_alpha_model_results_dict in all_alpha_model_results:
                if qid not in single_alpha_model_results_dict:
                    missing_in_some_alpha += 1
                    continue
                candidate_max_utility = single_alpha_model_results_dict[qid]["max-utility"]
                if candidate_max_utility > model_max_utility:
                    model_max_utility = candidate_max_utility
        else:
            model_eu: float = model_results_dict[qid]["EU"][utility_metric]
            model_eu = convert_to_higher_the_better(model_eu, upper_bound=4)

            # 'lower the better' metric should be converted to 'higher the better' metric
            # fallback baseline from current alpha's own minimum error
            model_min_error = model_results_dict[qid]["min-utility"]

            if qid in gold_results_dict:
                gold_max_utility = convert_to_higher_the_better(
                    gold_results_dict[qid]["min-utility"], upper_bound=4
                )
            else:
                gold_max_utility = convert_to_higher_the_better(
                    model_min_error, upper_bound=4
                )
                missing_in_gold += 1

            # get model's min error across all alphas
            for single_alpha_model_results_dict in all_alpha_model_results:
                if qid not in single_alpha_model_results_dict:
                    missing_in_some_alpha += 1
                    continue
                candidate_min_error = single_alpha_model_results_dict[qid]["min-utility"]
                if candidate_min_error < model_min_error:
                    model_min_error = candidate_min_error

            model_max_utility = convert_to_higher_the_better(
                model_min_error, upper_bound=4
            )

        # Normalizing model's EU
        max_utility = max(gold_max_utility, model_max_utility)
        try:
            normalized_eu: float = model_eu / max_utility
        except ZeroDivisionError:
            normalized_eu = 1.0

        # save the normalized EU
        save_dict[qid]["EU"][utility_metric] = normalized_eu

    # Save the normalized results file
    with open(SAVE_FP, "w") as f:
        json.dump(save_dict, f, indent=2)
    f.close()

    if missing_in_gold > 0 or missing_in_some_alpha > 0:
        print(
            "Warning: normalization used fallback due to partial qid overlap "
            f"(missing_in_gold={missing_in_gold}, missing_in_some_alpha={missing_in_some_alpha})."
        )


if __name__ == "__main__":
    # Example run:
    # python normalize_eu.py --retriever_name splade --generator_name flanT5XXL --lamp_num 4 --alpha 2
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
        required=True,
        help="Fairness control parameter in Plackett-Luce Sampling; alpha's result to normalize",
    )
    parser.add_argument(
        "--all_alphas",
        type=str,
        default="1,2,4,8",
        help="Comma-separated alpha values used to search for the max utility across runs",
    )
    parser.add_argument(
        "--input_suffix",
        type=str,
        default="",
        help="Optional input/output filename suffix, e.g. _mmr_deterministic",
    )
    args = parser.parse_args()

    main(args)
