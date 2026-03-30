"""
Hypothetical Oracle Retriever that always ranks useful items above non-useful ones with access to the utility labels
"""

import argparse
import json
import os
import pandas as pd

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def main(args):
    LAMP_NUM: int = args.lamp_num
    GENERATOR_NAME = args.generator_name
    OUTPUT_RANKING_DIR_PATH = os.path.join(
        CUR_DIR_PATH, "retrieval_results", GENERATOR_NAME, "gold"
    )
    os.makedirs(OUTPUT_RANKING_DIR_PATH, exist_ok=True)
    OUTPUT_RANKING_FP = os.path.join(OUTPUT_RANKING_DIR_PATH, f"{LAMP_NUM}.json")
    DELTA_FP = os.path.join(
        os.path.dirname(CUR_DIR_PATH),
        "utility_labels/eval_results",
        GENERATOR_NAME,
        f"{LAMP_NUM}_delta.tsv",
    )
    REL_FP = os.path.join(
        os.path.dirname(CUR_DIR_PATH),
        "data",
        f"lamp_utility_labels_{GENERATOR_NAME}",
        f"{LAMP_NUM}_relevance_mapping.tsv",
    )
    # loading this just to get the final qids
    FINAL_OUTPUT_DATA_FP = os.path.join(
        os.path.dirname(CUR_DIR_PATH),
        "data",
        f"lamp_utility_labels_{GENERATOR_NAME}",
        f"{LAMP_NUM}_user_dev_outputs.json",
    )
    with open(FINAL_OUTPUT_DATA_FP, "r") as f:
        final_data = json.load(f)

    final_qids: list[str] = []
    for entry_dict in final_data["golds"]:
        final_qids.append(entry_dict["id"])

    dtype_spec = {"qid": str, "pid": str}
    if os.path.exists(DELTA_FP):
        utility_df = pd.read_csv(DELTA_FP, sep="\t", dtype=dtype_spec)
        utility_df = utility_df[utility_df["qid"].isin(final_qids)]
        utility_df = utility_df[["qid", "pid", "delta"]].copy()
        utility_df["score"] = (utility_df["delta"] > 0).astype(int)
        sort_cols = ["score", "pid"]
    elif os.path.exists(REL_FP):
        utility_df = pd.read_csv(REL_FP, sep="\t", dtype=dtype_spec)
        utility_df = utility_df[utility_df["qid"].isin(final_qids)]
        utility_df = utility_df[["qid", "pid", "relevance_label"]].copy()
        utility_df = utility_df.rename(columns={"relevance_label": "score"})
        sort_cols = ["score", "pid"]
    else:
        raise FileNotFoundError(
            "Missing utility labels for gold retriever. Expected one of:\n"
            f" - {DELTA_FP}\n"
            f" - {REL_FP}"
        )

    # oracle retrieval: rank by binary utility score and tie-break by pid.
    rank_dict = dict()
    for qid in final_qids:
        qid_df = utility_df[utility_df["qid"] == qid]
        sorted_qid_df = qid_df.sort_values(by=sort_cols, ascending=[False, False])
        pid_list = sorted_qid_df["pid"].tolist()
        score_list = sorted_qid_df["score"].astype(int).tolist()
        qid_retrieval_results = []
        for pid, score in zip(pid_list, score_list):
            qid_retrieval_results.append((pid, score))
        rank_dict.update({qid: qid_retrieval_results})

    with open(OUTPUT_RANKING_FP, "w") as f:
        json.dump(rank_dict, f)
    f.close()


if __name__ == "__main__":
    # Example Run:
    # python retrieval/gold_retriever.py --generator_name flanT5Base --lamp_num 1
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lamp_num",
        type=int,
        help="LaMP number",
    )
    parser.add_argument(
        "--generator_name",
        type=str,
        default="flanT5XXL",
        help="Generator model nickname of HF model",
    )
    args = parser.parse_args()

    main(args)
