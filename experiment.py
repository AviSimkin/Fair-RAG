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


import csv
import os
import argparse
import warnings
import numpy as np
import json
import re
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

# Suppress non-critical warnings to reduce output noise
warnings.filterwarnings('ignore', category=FutureWarning, message='.*resume_download.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*pipelines sequentially on GPU.*')

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
from generator.lm_mlx import PromptLMMLX
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


def _save_ckpt_atomic(ckpt_fp: str, results: dict) -> None:
    """Atomically write a checkpoint file (survives machine kills on POSIX systems)."""
    tmp_fp = ckpt_fp + ".tmp"
    with open(tmp_fp, "w") as f:
        json.dump({"results": results}, f)
    os.replace(tmp_fp, ckpt_fp)


def _append_progress_row(progress_fp: str, row: list) -> None:
    """Append one row to the rolling progress CSV."""
    with open(progress_fp, "a", newline="") as f:
        csv.writer(f).writerow(row)


def _load_qids_from_file(fp: str) -> set[str]:
    """Load qids from a JSON list/object or a plain text file (one per line / comma-separated)."""
    with open(fp, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return set()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            if "qids" in parsed and isinstance(parsed["qids"], list):
                return {str(x).strip() for x in parsed["qids"] if str(x).strip()}
            return {str(k).strip() for k in parsed.keys() if str(k).strip()}
        if isinstance(parsed, list):
            return {str(x).strip() for x in parsed if str(x).strip()}
    except json.JSONDecodeError:
        pass

    qids: set[str] = set()
    for line in raw.splitlines():
        for token in line.split(","):
            qid = token.strip()
            if qid:
                qids.add(qid)
    return qids


def main(args):
    LAMP_NUM: int = args.lamp_num
    SPLIT_TYPE = args.lamp_split_type
    GENERATOR_NAME = args.generator_name
    model_info = models_info[GENERATOR_NAME]
    model_backend = model_info.get("backend", "hf")
    model_id = model_info["model_id"]
    model_kwargs = model_info.get("model_kwargs", {})
    allow_hf_download = bool(model_kwargs.get("allow_hf_download", False))

    tokenizer_source = model_id
    if model_backend == "mlx":
        direct_path = Path(model_id)
        if direct_path.exists():
            tokenizer_source = str(direct_path)
        else:
            try:
                tokenizer_source = str(
                    snapshot_download(
                        model_id,
                        local_files_only=(not allow_hf_download),
                    )
                )
            except Exception as exc:
                raise FileNotFoundError(
                    f"Model/tokenizer {model_id} is not available in local cache. "
                    "Download it once or set allow_hf_download=True."
                ) from exc
    else:
        # Prefer local cache to avoid unnecessary HF Hub requests/warnings.
        try:
            tokenizer_source = str(snapshot_download(model_id, local_files_only=True))
        except Exception:
            tokenizer_source = model_id

    tokenizer_local_only = bool(model_backend == "mlx" and not allow_hf_download)
    TOKENIZER = AutoTokenizer.from_pretrained(
        tokenizer_source,
        local_files_only=tokenizer_local_only,
    )
    # Use conservative token limit (20% buffer) to prevent overflow errors during tokenization
    TOKENIZER_MAX_LEN = max(1, int(TOKENIZER.model_max_length * 0.8))
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
    target_qids: set[str] | None = None
    if args.only_qids_file:
        target_qids = _load_qids_from_file(args.only_qids_file)
        if not target_qids:
            raise ValueError(f"No qids found in --only_qids_file: {args.only_qids_file}")
        print(
            f"[targeted] qid filter active: {len(target_qids)} qids "
            f"from {args.only_qids_file}",
            flush=True,
        )
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
    # --- Checkpoint path (same dir as final output, different filename) ---
    CHECKPOINT_FP = EXP_RESULTS_FP[:-5] + "_ckpt.json"

    # --- Organized run-log directory ---
    _nq_str = str(args.max_queries) if args.max_queries is not None else "all"
    _suffix_clean = output_suffix.lstrip("_").replace("_", "-") if output_suffix else "plain"
    _run_tag = getattr(args, "run_tag", "") or ""
    _run_tag_str = f"_{_run_tag}" if _run_tag else ""
    _run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _run_dir_name = (
        f"{_run_ts}_{base_retriever_name}_alpha{ALPHA}_{_suffix_clean}"
        f"_{GENERATOR_NAME}_lamp{LAMP_NUM}_nq{_nq_str}{_run_tag_str}"
    )
    RUN_LOG_DIR = os.path.join(CUR_DIR_PATH, "experiment_results", "runs", _run_dir_name)
    os.makedirs(RUN_LOG_DIR, exist_ok=True)

    _params_obj = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "retriever": RETRIEVER_NAME,
        "base_retriever": base_retriever_name,
        "generator": GENERATOR_NAME,
        "lamp_num": LAMP_NUM,
        "alpha": ALPHA,
        "output_suffix": output_suffix,
        "run_tag": _run_tag,
        "max_queries": _nq_str,
        "k": args.k,
        "n_samples": args.n_samples,
        "deterministic": force_deterministic,
        "mmr_lambda": args.mmr_lambda if is_mmr_retriever else None,
        "checkpoint_interval": args.checkpoint_interval,
        "only_qids_file": args.only_qids_file or None,
        "recompute_target_qids": bool(args.recompute_target_qids),
        "output_file": EXP_RESULTS_FP,
        "llm_outputs_file": None,
        "completed_at": None,
    }
    with open(os.path.join(RUN_LOG_DIR, "params.json"), "w") as _pf:
        json.dump(_params_obj, _pf, indent=2)

    PROGRESS_CSV_FP = os.path.join(RUN_LOG_DIR, "progress.csv")
    _PROGRESS_HEADER = [
        "timestamp", "query_count", "total_queries", "new_this_run",
        "retriever", "alpha", "suffix", "run_tag",
        "mean_ee_d", "mean_ee_r", "mean_eu",
    ]
    with open(PROGRESS_CSV_FP, "w", newline="") as _pcsv:
        csv.writer(_pcsv).writerow(_PROGRESS_HEADER)
    print(f"[run-log] {RUN_LOG_DIR}", flush=True)

    LLM_OUTPUTS_FP = None
    if args.save_llm_outputs:
        LLM_OUTPUTS_FP = os.path.join(RUN_LOG_DIR, "llm_outputs.jsonl")
        _params_obj["llm_outputs_file"] = LLM_OUTPUTS_FP
        with open(os.path.join(RUN_LOG_DIR, "params.json"), "w") as _pf:
            json.dump(_params_obj, _pf, indent=2)
        print(f"[run-log] llm outputs will be saved to {LLM_OUTPUTS_FP}", flush=True)

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
        tokenizer_model_name=tokenizer_source,
        k=K,
    )
    aip_func = lamp_handler.get_aip_func(lamp_num=LAMP_NUM)
    inputs_file_iterator = lamp_handler.get_inputs_file_iterator(lamp_number=LAMP_NUM)
    outputs_file_iterator = lamp_handler.get_outputs_file_iterator(lamp_number=LAMP_NUM)
    if model_backend == "mlx":
        qa_model = PromptLMMLX(
            model_name=GENERATOR_NAME,
            model_kwargs=model_kwargs,
        )
    elif args.multi_gpus:
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
    target_mode = target_qids is not None
    if target_mode and os.path.exists(EXP_RESULTS_FP):
        with open(EXP_RESULTS_FP, "r", encoding="utf-8") as _ef:
            _existing = json.load(_ef)
        if isinstance(_existing, dict):
            qid_results_map.update(_existing)
            print(
                f"[targeted] bootstrapped {len(_existing)} existing results from {EXP_RESULTS_FP}",
                flush=True,
            )
        if args.recompute_target_qids:
            _removed = 0
            for _qid in target_qids:
                if _qid in qid_results_map:
                    qid_results_map.pop(_qid, None)
                    _removed += 1
            print(
                f"[targeted] removed {_removed} target qids from existing results; "
                "they will be recomputed",
                flush=True,
            )
    # Iterate over qids
    max_queries = args.max_queries  # None means run all
    query_count = 0
    new_query_count = 0  # tracks only queries processed fresh in this run (not resumed)
    _run_ee_d = 0.0
    _run_ee_r = 0.0
    _run_eu = 0.0

    _resumed_count = 0
    # --- Resume from checkpoint if one exists ---
    if os.path.exists(CHECKPOINT_FP):
        with open(CHECKPOINT_FP, "r") as _cf:
            _ckpt_data = json.load(_cf)
        _ckpt_results = _ckpt_data.get("results", {})
        if isinstance(_ckpt_results, dict):
            qid_results_map.update(_ckpt_results)

        if target_mode and args.recompute_target_qids:
            for _qid in target_qids:
                qid_results_map.pop(_qid, None)

        _resumed_qids = set(_ckpt_results.keys()) if isinstance(_ckpt_results, dict) else set()
        if target_mode:
            _resumed_qids &= target_qids
        _resumed_qids = {_q for _q in _resumed_qids if _q in qid_results_map}
        _resumed_count = len(_resumed_qids)
        if _resumed_count > 0:
            for _qid in _resumed_qids:
                _v = qid_results_map[_qid]
                _run_ee_d += _v["EE"].get("disparity", 0.0)
                _run_ee_r += _v["EE"].get("relevance", 0.0)
                _eu_dict = _v.get("EU", {})
                _run_eu += float(next(iter(_eu_dict.values()))) if _eu_dict else 0.0
            print(
                f"[checkpoint] Resuming: {_resumed_count} queries already done. "
                f"avg EE-R so far: {_run_ee_r / _resumed_count:.4f}",
                flush=True,
            )
    _total_label = (
        f"target:{len(target_qids)}"
        if target_mode
        else (str(max_queries) if max_queries is not None else "all")
    )
    for input_entry, output_entry in zip(inputs_file_iterator, outputs_file_iterator):
        if max_queries is not None and query_count >= max_queries:
            break
        query_count += 1
        assert input_entry["id"] == output_entry["id"]
        qid: str = input_entry["id"]
        if target_mode and qid not in target_qids:
            continue
        # Resume: skip queries already processed and stored in checkpoint
        if qid in qid_results_map:
            continue
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
        scores = np.array(scores, dtype=float)
        if RETRIEVER_NAME != "gold":
            # ensure no negative scores (since we are going to apply ALPHA)
            min_value = float(scores.min())
            if min_value < 0:
                # rescale the scores to have minimum value of 0
                scores = scores - min_value
            min_value = float(scores.min())
            max_value = float(scores.max())
            denom = max_value - min_value
            # Min-Max Normalization, followed by scaling to [1, 2]
            if not np.isfinite(denom) or denom <= 0:
                # Degenerate score vector (all equal or invalid): use uniform weights.
                print(
                    f"[warn] degenerate retriever scores for qid={qid}; using uniform scores",
                    flush=True,
                )
                scores = np.ones_like(scores, dtype=float)
            else:
                scores = (scores - min_value) / denom
            scores = scores + 1  # rescale to [1, 2] to amplify the effect of ALPHA
        else:  # oracle retriever
            # retrieval scores for oracle retriever are in binary (either 0 or 1)
            # amplify positive label (1) to make ALPHA effect bigger and make sure we do not have EE-R less than 1 for lower ALPHA
            scores = np.where(scores > 0, 10, 0)

        # Apply ALPHA as a temperature parameter
        scores = scores**ALPHA

        ## Perform ranking generation
        if force_deterministic:
            # For MMR, retrieval_results_qid is already re-ordered by MMR score.
            # Keep that order instead of re-sorting by original retriever scores.
            if is_mmr_retriever:
                deterministic_ranking = np.arange(len(retrieval_results_qid))[:k]
            else:
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
                # Defensive: ensure we never exceed actual model limit
                token_count = len(TOKENIZER.tokenize(final_prompt))
                if token_count > TOKENIZER.model_max_length:
                    print(f"[warn] prompt token count {token_count} exceeds model max {TOKENIZER.model_max_length}; aggressive re-trim", flush=True)
                    final_prompt = trim_sentence_by_token_len(
                        final_prompt, tokenizer=TOKENIZER, max_tok_len=int(TOKENIZER.model_max_length * 0.7)
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

        if LLM_OUTPUTS_FP is not None:
            llm_row = {
                "qid": qid,
                "question": entry_question,
                "target": entry_target,
                "retriever": RETRIEVER_NAME,
                "base_retriever": base_retriever_name,
                "alpha": ALPHA,
                "output_suffix": output_suffix,
                "deterministic": bool(force_deterministic),
                "n_samples_requested": int(N_SAMPLES),
                "n_samples_effective": int(effective_n_samples),
                "predictions": preds,
                "metric_name": metric_name,
                "metric_scores": metric_scores,
                "expected_utility": expected_utility,
            }
            with open(LLM_OUTPUTS_FP, "a", encoding="utf-8") as _lf:
                _lf.write(json.dumps(llm_row, ensure_ascii=False) + "\n")

        # Update running averages and print progress every 10 queries
        _run_ee_d += ee_results.get("disparity", 0.0)
        _run_ee_r += ee_results.get("relevance", 0.0)
        _run_eu += expected_utility
        new_query_count += 1
        _total_processed = _resumed_count + new_query_count
        _print_every = args.print_interval if args.print_interval is not None else args.checkpoint_interval
        if new_query_count % args.checkpoint_interval == 0:
            if new_query_count % _print_every == 0:
                print(
                    f"[{_total_processed}/{_total_label}] "
                    f"avg EE-D: {_run_ee_d / _total_processed:.4f} | "
                    f"avg EE-R: {_run_ee_r / _total_processed:.4f} | "
                    f"avg EU ({metric_name}): {_run_eu / _total_processed:.4f}",
                    flush=True,
                )
            # Save checkpoint so we can resume if the machine dies
            _save_ckpt_atomic(CHECKPOINT_FP, qid_results_map)
            # Append a row to the rolling progress CSV
            _append_progress_row(PROGRESS_CSV_FP, [
                datetime.now().isoformat(timespec="seconds"),
                _total_processed,
                _total_label,
                new_query_count,
                base_retriever_name,
                ALPHA,
                output_suffix or "plain",
                _run_tag,
                f"{_run_ee_d / _total_processed:.6f}",
                f"{_run_ee_r / _total_processed:.6f}",
                f"{_run_eu / _total_processed:.6f}",
            ])

    _total_processed = _resumed_count + new_query_count
    if _total_processed > 0 and new_query_count % args.checkpoint_interval != 0:
        _print_every = args.print_interval if args.print_interval is not None else args.checkpoint_interval
        print(
            f"[{_total_processed}/{_total_label}] FINAL "
            f"avg EE-D: {_run_ee_d / _total_processed:.4f} | "
            f"avg EE-R: {_run_ee_r / _total_processed:.4f} | "
            f"avg EU ({metric_name}): {_run_eu / _total_processed:.4f}",
            flush=True,
        )
        # Final progress row
        _append_progress_row(PROGRESS_CSV_FP, [
            datetime.now().isoformat(timespec="seconds"),
            _total_processed,
            _total_label,
            new_query_count,
            base_retriever_name,
            ALPHA,
            output_suffix or "plain",
            _run_tag,
            f"{_run_ee_d / _total_processed:.6f}",
            f"{_run_ee_r / _total_processed:.6f}",
            f"{_run_eu / _total_processed:.6f}",
        ])

    # Write experiment results for the LAMP_NUM
    with open(EXP_RESULTS_FP, "w") as f:
        json.dump(qid_results_map, f, indent=2)
        f.close()

    # Remove checkpoint (run completed successfully) and finalize run log
    if os.path.exists(CHECKPOINT_FP):
        os.remove(CHECKPOINT_FP)
    _params_obj["completed_at"] = datetime.now().isoformat(timespec="seconds")
    _params_obj["total_queries_processed"] = _total_processed
    with open(os.path.join(RUN_LOG_DIR, "params.json"), "w") as _pf:
        json.dump(_params_obj, _pf, indent=2)
    print(f"[run-log] completed → {RUN_LOG_DIR}", flush=True)


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
    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help=(
            "Human-readable label for this run (e.g. 'balanced', 'weak'). "
            "Used in the run-log folder name."
        ),
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10,
        help="Save a checkpoint and log progress every N newly processed queries.",
    )
    parser.add_argument(
        "--print_interval",
        type=int,
        default=None,
        help=(
            "Print progress to stdout every N newly processed queries. "
            "Defaults to checkpoint_interval when not set. "
            "Set higher (e.g. 40) for long runs to reduce noise."
        ),
    )
    parser.add_argument(
        "--only_qids_file",
        type=str,
        default="",
        help=(
            "Optional path to qids list (json/txt). If set, only those qids are processed. "
            "Useful for targeted recomputation."
        ),
    )
    parser.add_argument(
        "--recompute_target_qids",
        action="store_true",
        help=(
            "When used with --only_qids_file, remove target qids from existing/checkpoint "
            "results before running so they are recomputed and merged back."
        ),
    )
    parser.add_argument(
        "--save_llm_outputs",
        action="store_true",
        help=(
            "Persist raw per-query LLM outputs to experiment_results/runs/<run>/llm_outputs.jsonl "
            "for later inspection or cleaning."
        ),
    )
    args = parser.parse_args()

    main(args)
