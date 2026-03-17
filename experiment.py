# Main experiment
"""
1. Load/Set
    1. LaMP Number
    2. Retrieval results from a deterministic ranker (either BM25 or Contriever)
    3. Alpha (sampling temperature parameter)

2. For each query:
    1. Get precomputed deterministic retrieval results (profiles)
    2. Perform plackett-luce sampling (parameterized by alpha) and generate N_SAMPLES rankings with cutoff (k) 
    3. Compute Expected Exposure (EE-D, EE-R, EE-L) of the stochastic rankings
        1. Load relevance labels (trec_rel_file)
        2. Make trec_top_file
        3. Pass parameters to EE package and get EE results for this query
    For each ranking (\sigma) in the sampled_rankings (\Sigma):
        1. Construct prompts for an LLM (e.g., flanT5XXL) 
        2. Make inferences using the prompt-augmented LLM
        3. Evaluate using the gold string label (e.g., data/lamp1000_useful_flanT5XXL/xx_outputs.json) 
            and the string utility metrics (eval.lamp_metrics)
    4. Compute Expected Utility (Unnormalized EU)
"""


import os
import argparse
import numpy as np
import json
import re
from transformers import AutoTokenizer

from utils import (
    models_info,
    trim_sentence_by_token_len,
    make_trec_top_file_for_single_qid,
    make_trec_rel_file_for_single_qid,
)
from eval.lamp_metrics import (
    get_metric_fn_accuracy,
    get_metric_fn_mae,
    get_metric_fn_rouge_L,
)
from perturbation import plackettluce as pl

from data.lamp_handler import LaMPHandler
from expected_exposure import expeval
from generator.lm import PromptLM
from generator.lm_distributed_inference import PromptLMDistributedInference
import logging


logger = logging.getLogger()
CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

ee_eval_params = {
    "umType": "rbp",
    "umPatience": 1,  # we assume that a machine-user browse items with patience 1 until it sees k items.
    "umUtility": 0.5,  # won't matter for rbp
    "binarize": False,
    "groupEvaluation": False,
    "complete": False,
    "normalize": True,
    "relfn": "",  # will be dynamically filled while this script is being run
    "topfn": "",  # will be dynamically filled while this script is being run
}


def _tokenize_for_mmr(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _profile_to_text(profile: dict) -> str:
    text_parts: list[str] = []
    for k, v in profile.items():
        if k == "id":
            continue
        if isinstance(v, str):
            text_parts.append(v)
    return " ".join(text_parts)


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a.intersection(b)) / len(a.union(b))


def rerank_with_mmr(
    retrieval_results_qid: list[list],
    query_text: str,
    profile_lookup: dict[str, dict],
    mmr_lambda: float,
) -> list[list]:
    """Re-rank candidates with MMR using query relevance + lexical diversity."""
    if not retrieval_results_qid:
        return retrieval_results_qid

    mmr_lambda = min(1.0, max(0.0, mmr_lambda))
    pids = [pid_score[0] for pid_score in retrieval_results_qid]
    scores = np.array([float(pid_score[1]) for pid_score in retrieval_results_qid])

    s_min = scores.min()
    s_max = scores.max()
    if s_max > s_min:
        rel_scores = (scores - s_min) / (s_max - s_min)
    else:
        rel_scores = np.ones_like(scores)

    query_tokens = _tokenize_for_mmr(query_text)
    doc_tokens: list[set[str]] = []
    for pid in pids:
        profile = profile_lookup.get(pid, {})
        doc_tokens.append(_tokenize_for_mmr(_profile_to_text(profile)))

    selected: list[int] = []
    remaining = set(range(len(pids)))
    while remaining:
        best_idx = None
        best_score = None
        for idx in remaining:
            relevance = float(rel_scores[idx])
            if not selected:
                diversity_penalty = 0.0
            else:
                diversity_penalty = max(
                    _jaccard_similarity(doc_tokens[idx], doc_tokens[sel_idx])
                    for sel_idx in selected
                )
            mmr_score = mmr_lambda * relevance - (1.0 - mmr_lambda) * diversity_penalty
            if best_score is None or mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [retrieval_results_qid[i] for i in selected]


def get_labels(lamp_num):
    if lamp_num == 1:
        return ["[1]", "[2]"]
    elif lamp_num == 2:
        return [
            "sci-fi",
            "based on a book",
            "comedy",
            "action",
            "twist ending",
            "dystopia",
            "dark comedy",
            "classic",
            "psychology",
            "fantasy",
            "romance",
            "thought-provoking",
            "social commentary",
            "violence",
            "true story",
        ]
    elif lamp_num == 3:
        return ["1", "2", "3", "4", "5"]
    else:
        raise ValueError(f"LaMP {lamp_num} is not classification task")


def main(args):
    LAMP_NUM: int = args.lamp_num
    SPLIT_TYPE = args.lamp_split_type
    GENERATOR_NAME = args.generator_name
    TOKENIZER = AutoTokenizer.from_pretrained(models_info[GENERATOR_NAME]["model_id"])
    TOKENIZER_MAX_LEN = TOKENIZER.model_max_length
    RETRIEVER_NAME = args.retriever_name  # deterministic retriever
    is_mmr_retriever = RETRIEVER_NAME == "mmr"
    base_retriever_name = args.mmr_base_retriever if is_mmr_retriever else RETRIEVER_NAME
    DATASET = args.dataset
    ALPHA: int = args.alpha
    output_suffix = args.output_suffix.strip()
    if output_suffix and not output_suffix.startswith("_"):
        output_suffix = f"_{output_suffix}"
    K: int = args.k
    N_SAMPLES: int = args.n_samples
    force_deterministic = args.deterministic_ranking or is_mmr_retriever
    effective_n_samples = 1 if force_deterministic else N_SAMPLES
    if force_deterministic and N_SAMPLES != 1:
        print(
            f"[deterministic] forcing effective n_samples=1 (received n_samples={N_SAMPLES})",
            flush=True,
        )
    REL_MAPPING_FP = os.path.join(
        CUR_DIR_PATH,
        "data",
        f"lamp_utility_labels_{GENERATOR_NAME}",
        f"{LAMP_NUM}_relevance_mapping.tsv",
    )
    EXP_RESULTS_DIR_PATH = os.path.join(
        CUR_DIR_PATH,
        "experiment_results",
        GENERATOR_NAME,
        f"lamp{LAMP_NUM}",
        RETRIEVER_NAME,
    )
    os.makedirs(EXP_RESULTS_DIR_PATH, exist_ok=True)
    EXP_RESULTS_FP = os.path.join(
        EXP_RESULTS_DIR_PATH,
        f"alpha_{ALPHA}{output_suffix}.json",
    )
    del EXP_RESULTS_DIR_PATH
    RETRIEVAL_RESULTS_FP = os.path.join(
        CUR_DIR_PATH,
        "retrieval",
        "retrieval_results",
        GENERATOR_NAME,
        base_retriever_name,
        f"{LAMP_NUM}.json",
    )
    with open(RETRIEVAL_RESULTS_FP, "r") as f:
        retrieval_results: dict = json.load(f)
    f.close()
    del RETRIEVAL_RESULTS_FP

    lamp_handler = LaMPHandler(
        lamp_dir_name=f"lamp_utility_labels_{GENERATOR_NAME}",
        split_type=SPLIT_TYPE,
        tokenizer_model_name=models_info[GENERATOR_NAME]["model_id"],
        k=K,
    )
    aip_func = lamp_handler.get_aip_func(lamp_num=LAMP_NUM)
    inputs_file_iterator = lamp_handler.get_inputs_file_iterator(lamp_number=LAMP_NUM)
    outputs_file_iterator = lamp_handler.get_outputs_file_iterator(lamp_number=LAMP_NUM)
    if args.multi_gpus:
        qa_model = PromptLMDistributedInference(model_name=GENERATOR_NAME)
    else:
        qa_model = PromptLM(model_name=GENERATOR_NAME)

    # set corresponding metric function for a LaMP task
    if LAMP_NUM in {1, 2}:
        metric_name = "acc"
        metric_fn = get_metric_fn_accuracy(get_labels(LAMP_NUM))
    elif LAMP_NUM in {3}:
        metric_name = "mae"
        metric_fn = get_metric_fn_mae()
    else:
        metric_name = "rouge-l"
        metric_fn = get_metric_fn_rouge_L()

    # qid_results_map:
    # {
    #  qid: { "EE": {'disparity': float, 'relevance': float, 'difference': float},
    #         "EU": {'rouge-l': float}
    #       }
    # }
    qid_results_map = dict()
    # Iterate over qids
    max_queries = args.max_queries  # None means run all
    query_count = 0
    _run_ee_d = 0.0
    _run_ee_r = 0.0
    _run_eu = 0.0
    _total_label = str(max_queries) if max_queries is not None else "all"
    for input_entry, output_entry in zip(inputs_file_iterator, outputs_file_iterator):
        if max_queries is not None and query_count >= max_queries:
            break
        query_count += 1
        assert input_entry["id"] == output_entry["id"]
        qid: str = input_entry["id"]
        entry_question: str = input_entry["input"]
        entry_target: str = output_entry["output"]  # gold label
        qid_results_map.update(
            {qid: {"EE": {}, "EU": {}, "max-utility": None, "min-utility": None}}
        )

        retrieval_results_qid = retrieval_results[qid]
        if is_mmr_retriever:
            profile_lookup = {x["id"]: x for x in input_entry["profile"]}
            retrieval_results_qid = rerank_with_mmr(
                retrieval_results_qid,
                query_text=entry_question,
                profile_lookup=profile_lookup,
                mmr_lambda=args.mmr_lambda,
            )

        k = len(retrieval_results_qid) if K == -1 else K
        scores = [pid_score_pair[1] for pid_score_pair in retrieval_results_qid]
        scores = np.array(scores)
        min_value = scores.min()
        max_value = scores.max()
        if RETRIEVER_NAME != "gold":
            # ensure no negative scores (since we are going to apply ALPHA)
            if min_value < 0:
                # rescale the scores to have minimum value of 0
                scores = scores - min_value
            # Min-Max Normalization, followed by scaling to [1, 2]
            scores = (scores - min_value) / (max_value - min_value)
            scores = scores + 1  # rescale to [1, 2] to amplify the effect of ALPHA
        else:  # oracle retriever
            # retrieval scores for oracle retriever are in binary (either 0 or 1)
            # amplify positive label (1) to make ALPHA effect bigger and make sure we do not have EE-R less than 1 for lower ALPHA
            scores = np.where(scores > 0, 10, 0)

        # Apply ALPHA as a temperature parameter
        scores = scores**ALPHA

        ## Perform ranking generation
        if force_deterministic:
            # Deterministic baseline: always use the same top-k ranking.
            deterministic_ranking = np.argsort(-scores, kind="mergesort")[:k]
            sampled_rankings = deterministic_ranking.reshape(1, -1)
        else:
            pl_result = pl.gumbel_sample_rankings(
                scores, effective_n_samples, cutoff=k, doc_prob=False
            )
            sampled_rankings = pl_result[0]

        ## Compute Expected Exposure
        # make trec_top_file and trec_rel_file
        trec_top_file_fp = make_trec_top_file_for_single_qid(
            qid=qid,
            rankings=sampled_rankings,
            retrieval_results=retrieval_results_qid,
            run_id="main_exp",
        )
        trec_rel_file_fp = make_trec_rel_file_for_single_qid(
            qid=qid, relevance_mapping_fp=REL_MAPPING_FP
        )
        ee_eval_params["topfn"] = trec_top_file_fp
        ee_eval_params["relfn"] = trec_rel_file_fp
        # run expected exposure evaluation
        # returns {'disparity': float, 'relevance': float, 'difference': float}
        ee_results: dict = expeval.run(parameters=ee_eval_params, k=k)
        qid_results_map[qid]["EE"] = ee_results  # update qid_results_map's EE
        if args.remove_temp_files:
            os.remove(trec_top_file_fp)
            os.remove(trec_rel_file_fp)

        ## Iterate through Sampled Rankings (\Sigma) to get Expected Utility
        # In this experiement, the LLM is deterministic, so we can cache results.
        ranking_pred_map = dict()
        preds: list[str] = []
        targets: list[str] = [entry_target for _ in range(effective_n_samples)]
        for ranking in sampled_rankings:
            # ranking is a list of indices of original ranking from deterministic ranker
            if str(ranking) not in ranking_pred_map:
                top_profiles: list[list] = [retrieval_results_qid[i] for i in ranking]
                pids: list = [x[0] for x in top_profiles]
                selected_profiles: list[dict] = lamp_handler.find_profiles_by_pids(
                    LAMP_NUM, qid, pids
                )
                final_prompt = aip_func(
                    question=entry_question, profiles=selected_profiles
                )
                final_prompt = trim_sentence_by_token_len(
                    final_prompt, tokenizer=TOKENIZER, max_tok_len=TOKENIZER_MAX_LEN
                )
                pred = qa_model.answer_question(final_prompt=final_prompt).strip()
                pred = (
                    "<empty>"
                    if pred == "" or (all(char in {".", " "} for char in pred))
                    else pred
                )
                # update mapping
                ranking_pred_map.update({str(ranking): pred})
            else:
                pred = ranking_pred_map[str(ranking)]
            preds.append(pred)

        # Evaluation of N_SAMPLE pred-target pairs and get one EU
        try:
            assert effective_n_samples == len(preds) == len(targets)
        except:
            logger.error(f"Evaluation length mismatch: skipping qid: {qid}", flush=True)
        metric_scores: list = metric_fn(preds, targets)
        expected_utility = float(sum(metric_scores) / len(metric_scores))
        qid_results_map[qid]["EU"] = {metric_name: expected_utility}
        qid_results_map[qid]["max-utility"] = max(metric_scores)
        qid_results_map[qid]["min-utility"] = min(metric_scores)

        # Update running averages and print progress every 10 queries
        _run_ee_d += ee_results.get("disparity", 0.0)
        _run_ee_r += ee_results.get("relevance", 0.0)
        _run_eu += expected_utility
        if query_count % 10 == 0:
            print(
                f"[{query_count}/{_total_label}] "
                f"avg EE-D: {_run_ee_d / query_count:.4f} | "
                f"avg EE-R: {_run_ee_r / query_count:.4f} | "
                f"avg EU ({metric_name}): {_run_eu / query_count:.4f}",
                flush=True,
            )

    if query_count % 10 != 0:
        print(
            f"[{query_count}/{_total_label}] FINAL "
            f"avg EE-D: {_run_ee_d / query_count:.4f} | "
            f"avg EE-R: {_run_ee_r / query_count:.4f} | "
            f"avg EU ({metric_name}): {_run_eu / query_count:.4f}",
            flush=True,
        )

    # Write experiment results for the LAMP_NUM
    with open(EXP_RESULTS_FP, "w") as f:
        json.dump(qid_results_map, f, indent=2)
        f.close()


if __name__ == "__main__":
    # Example run:
    #
    # 1) single GPU
    # python experiment.py --retriever_name splade --generator_name flanT5Base --lamp_num 7 --alpha 2
    #
    # 2) multi-GPU
    # accelerate launch --gpu_ids 0,1 --num_processes 1 --main_process_port 29500 \
    #   experiment.py --multi_gpus --retriever_name splade --generator_name flanT5XXL \
    #   --lamp_num 7 --alpha 1
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lamp_num",
        type=int,
        required=True,
        help="LaMP number",
    )
    parser.add_argument(
        "--lamp_split_type",
        type=str,
        default="user",
        help="data split type of LaMP: either 'user' or 'time'",
    )
    parser.add_argument(
        "--generator_name",
        type=str,
        default="flanT5XXL",
        help="Generator model nickname of HF model",
    )
    parser.add_argument(
        "--retriever_name",
        type=str,
        required=True,
        help="Deterministic retriever model nickname. bm25; contriever",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="lamp_utility_labels",
        help="Filtered Dataset Directory Name",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        required=True,
        help="Fairness control parameter in Plackett-Luce Sampling",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k retrieval; set to -1 (default) for all documents.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples for Plackett-Luce sampling",
    )
    parser.add_argument(
        "--remove_temp_files",
        action="store_true",
        help="Remove temporary trec files made for EE evaluation",
    )
    parser.add_argument(
        "--multi_gpus",
        action="store_true",
        help="Use multiple GPUs for distributed inference",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Limit number of queries (useful for quick testing). None means all.",
    )
    parser.add_argument(
        "--deterministic_ranking",
        action="store_true",
        help=(
            "Use deterministic top-k ranking from retriever scores (no Gumbel sampling). "
            "Useful for deterministic baseline runs."
        ),
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help=(
            "Optional suffix added to result filename. "
            "For example, alpha_1_deterministic.json"
        ),
    )
    parser.add_argument(
        "--mmr_base_retriever",
        type=str,
        default="bm25",
        help=(
            "Base retriever candidate list used by MMR mode. "
            "Only used when --retriever_name mmr"
        ),
    )
    parser.add_argument(
        "--mmr_lambda",
        type=float,
        default=0.7,
        help="MMR lambda in [0,1]. Higher favors relevance, lower favors diversity.",
    )
    args = parser.parse_args()

    main(args)
