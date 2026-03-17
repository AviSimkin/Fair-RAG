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
from transformers import AutoTokenizer
from datetime import datetime
import warnings

# Suppress transformers warnings
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

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
    "normalize": True,  # EE metrics use theoretical bounds for normalization
    "relfn": "",  # will be dynamically filled while this script is being run
    "topfn": "",  # will be dynamically filled while this script is being run
}


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
    DATASET = args.dataset
    ALPHA: int = args.alpha
    K: int = args.k
    N_SAMPLES: int = args.n_samples
    MAX_QUERIES: int = args.max_queries
    QUERY_BATCH_SIZE: int = args.query_batch_size
    RUN_ID: str = args.run_id
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
    
    # Build log file name with run_id to avoid overwriting
    if RUN_ID:
        log_suffix = f"alpha_{ALPHA}_{RUN_ID}.log"
    else:
        log_suffix = f"alpha_{ALPHA}.log"
    
    # Set up logging with immediate flush (save to both locations: experiment_results and logs/)
    log_file = os.path.join(EXP_RESULTS_DIR_PATH, log_suffix)
    
    # Also create log in tracked logs/ directory
    logs_dir = os.path.join(CUR_DIR_PATH, "logs", GENERATOR_NAME, f"lamp{LAMP_NUM}", RETRIEVER_NAME)
    os.makedirs(logs_dir, exist_ok=True)
    tracked_log_file = os.path.join(logs_dir, log_suffix)
    
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    
    # Add handlers for both locations
    for log_path in [log_file, tracked_log_file]:
        fh = logging.FileHandler(log_path, mode='w')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        fh.flush()  # Ensure handler flushes
        logger.addHandler(fh)
    
    # Log experiment configuration
    logger.info(f"{'='*60}")
    logger.info(f"Experiment Configuration")
    logger.info(f"{'='*60}")
    logger.info(f"LaMP Number: {LAMP_NUM}")
    logger.info(f"Generator: {GENERATOR_NAME}")
    logger.info(f"Retriever: {RETRIEVER_NAME}")
    logger.info(f"Alpha: {ALPHA}")
    logger.info(f"K (top-k): {K}")
    logger.info(f"N Samples: {N_SAMPLES}")
    if MAX_QUERIES:
        logger.info(f"Max Queries: {MAX_QUERIES}")
    logger.info(f"Query Batch Size: {QUERY_BATCH_SIZE}")
    
    EXP_RESULTS_FP = os.path.join(EXP_RESULTS_DIR_PATH, f"alpha_{ALPHA}.json")
    del EXP_RESULTS_DIR_PATH
    RETRIEVAL_RESULTS_FP = os.path.join(
        CUR_DIR_PATH,
        "retrieval",
        "retrieval_results",
        GENERATOR_NAME,
        RETRIEVER_NAME,
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
    
    # Update logging with metric name
    logger.info(f"Metric: {metric_name}")
    logger.info(f"{'='*60}\n")

    # qid_results_map:
    # {
    #  qid: { "EE": {'disparity': float, 'relevance': float, 'difference': float},
    #         "EU": {'rouge-l': float}
    #       }
    # }
    qid_results_map = dict()
    
    # Convert iterators to lists to get total count for progress reporting
    inputs_list = list(inputs_file_iterator)
    outputs_list = list(outputs_file_iterator)
    original_total = len(inputs_list)
    total_queries = original_total
    
    # Apply max_queries limit if specified
    if MAX_QUERIES and MAX_QUERIES < total_queries:
        inputs_list = inputs_list[:MAX_QUERIES]
        outputs_list = outputs_list[:MAX_QUERIES]
        total_queries = MAX_QUERIES
        logger.info(f"Limiting to {MAX_QUERIES} queries (original dataset had {original_total} queries)")
    
    print(f"Processing {total_queries} queries...", flush=True)
    
    # Iterate over qids
    for query_idx, (input_entry, output_entry) in enumerate(zip(inputs_list, outputs_list), 1):
        assert input_entry["id"] == output_entry["id"]
        qid: str = input_entry["id"]
        if query_idx % QUERY_BATCH_SIZE == 0 or query_idx == total_queries:
            progress_msg = f"Progress: {query_idx}/{total_queries} queries completed"
            print(f"  {progress_msg}", flush=True)
            logger.info(progress_msg)
            # Flush to ensure progress is saved even if process is interrupted
            for handler in logger.handlers:
                handler.flush()
        entry_question: str = input_entry["input"]
        entry_target: str = output_entry["output"]  # gold label
        qid_results_map.update(
            {qid: {"EE": {}, "EU": {}, "max-utility": None, "min-utility": None}}
        )

        k = len(retrieval_results[qid]) if K == -1 else K
        scores = [pid_score_pair[1] for pid_score_pair in retrieval_results[qid]]
        scores = np.array(scores, dtype=float)
        if RETRIEVER_NAME != "gold":
            # Min-Max normalization followed by scaling to [1, 2].
            # If all retrieval scores are identical, keep a uniform score vector.
            min_value = scores.min()
            max_value = scores.max()
            if max_value == min_value:
                scores = np.ones_like(scores)
            else:
                scores = (scores - min_value) / (max_value - min_value)
                scores = scores + 1  # rescale to [1, 2] to amplify the effect of ALPHA
        else:  # oracle retriever
            # retrieval scores for oracle retriever are in binary (either 0 or 1)
            # amplify positive label (1) to make ALPHA effect bigger and make sure we do not have EE-R less than 1 for lower ALPHA
            scores = np.where(scores > 0, 10, 0)

        # Apply ALPHA as a temperature parameter
        scores = scores**ALPHA

        ## Perform PL sampling
        pl_result = pl.gumbel_sample_rankings(
            scores, N_SAMPLES, cutoff=k, doc_prob=False
        )
        sampled_rankings = pl_result[0]

        ## Compute Expected Exposure
        # make trec_top_file and trec_rel_file
        trec_top_file_fp = make_trec_top_file_for_single_qid(
            qid=qid,
            rankings=sampled_rankings,
            retrieval_results=retrieval_results[qid],
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
        targets: list[str] = [entry_target for _ in range(N_SAMPLES)]
        for ranking in sampled_rankings:
            # ranking is a list of indices of original ranking from deterministic ranker
            if str(ranking) not in ranking_pred_map:
                top_profiles: list[list] = [retrieval_results[qid][i] for i in ranking]
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
            assert N_SAMPLES == len(preds) == len(targets)
        except:
            logger.error(f"Evaluation length mismatch: skipping qid: {qid}")
        metric_scores: list = metric_fn(preds, targets)
        expected_utility = float(sum(metric_scores) / len(metric_scores))
        qid_results_map[qid]["EU"] = {metric_name: expected_utility}
        qid_results_map[qid]["max-utility"] = max(metric_scores)
        qid_results_map[qid]["min-utility"] = min(metric_scores)

    # Print and log summary statistics
    print(f"\nCompleted {total_queries} queries.", flush=True)
    logger.info(f"\nExperiment Results Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total queries processed: {total_queries}")
    
    if qid_results_map:
        ee_d_vals = [v["EE"]["disparity"] for v in qid_results_map.values()]
        ee_r_vals = [v["EE"]["relevance"] for v in qid_results_map.values()]
        eu_vals = [v["EU"][metric_name] for v in qid_results_map.values()]
        
        ee_d_mean = float(np.mean(ee_d_vals))
        ee_d_std = float(np.std(ee_d_vals))
        ee_r_mean = float(np.mean(ee_r_vals))
        ee_r_std = float(np.std(ee_r_vals))
        eu_mean = float(np.mean(eu_vals))
        eu_std = float(np.std(eu_vals))
        
        print(f"Summary - EE-D (Disparity): mean={ee_d_mean:.4f}, std={ee_d_std:.4f}", flush=True)
        print(f"Summary - EE-R (Relevance): mean={ee_r_mean:.4f}, std={ee_r_std:.4f}", flush=True)
        print(f"Summary - Expected Utility: mean={eu_mean:.4f}, std={eu_std:.4f}", flush=True)
        
        logger.info(f"EE-D (Disparity) - Mean: {ee_d_mean:.6f}, Std: {ee_d_std:.6f}")
        logger.info(f"EE-R (Relevance) - Mean: {ee_r_mean:.6f}, Std: {ee_r_std:.6f}")
        logger.info(f"Expected Utility - Mean: {eu_mean:.6f}, Std: {eu_std:.6f}")
    
    logger.info(f"{'='*60}")
    logger.info(f"Results saved to: {EXP_RESULTS_FP}")
    logger.info(f"Tracked log saved to: {tracked_log_file} (committed to git)")
    
    # Flush before saving results
    for handler in logger.handlers:
        handler.flush()

    # Write experiment results for the LAMP_NUM
    with open(EXP_RESULTS_FP, "w") as f:
        json.dump(qid_results_map, f, indent=2)
        f.close()
    
    logger.info(f"Experiment completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # Final flush to ensure all logs are written
    for handler in logger.handlers:
        handler.flush()


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
        help="Maximum number of queries to process (None = all)",
    )
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=10,
        help="Report stats every N queries",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="",
        help="Run ID for timestamped logging (to avoid overwriting)",
    )
    args = parser.parse_args()

    main(args)
