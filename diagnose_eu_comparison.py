"""
Diagnostic: compare EU across models on the first 10 queries of LaMP-4.

For each model we load its generator, BM25 retrieval, and data, then run
a single deterministic inference per query. Output is a readable table showing:
  - query text (the article body snippet to headline)
  - gold label (the expected headline)
  - each model's prediction
  - each model's rouge-L score
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils import models_info, trim_sentence_by_token_len
from data.lamp_handler import LaMPHandler
from eval.lamp_metrics import get_metric_fn_rouge_L
from generator.lm import PromptLM
from generator.lm_mlx import PromptLMMLX

# ── Configuration ────────────────────────────────────────────────────────────
N_QUERIES = 10
LAMP_NUM = 4
K = 5
MAX_DISPLAY_LEN = 120  # characters – truncate long texts in the table

MODELS = [
    "flanT5Small",
    "lfm25MLX12B4bit",
    "qwen35MLX4B4bit",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_tokenizer_source(model_key: str) -> str:
    info = models_info[model_key]
    model_id = info["model_id"]
    backend = info.get("backend", "hf")
    if backend == "mlx":
        try:
            return str(snapshot_download(model_id, local_files_only=True))
        except Exception:
            return str(snapshot_download(model_id, local_files_only=False))
    else:
        try:
            return str(snapshot_download(model_id, local_files_only=True))
        except Exception:
            return model_id


def _build_generator(model_key: str):
    info = models_info[model_key]
    backend = info.get("backend", "hf")
    if backend == "mlx":
        return PromptLMMLX(model_name=model_key, model_kwargs=info.get("model_kwargs", {}))
    return PromptLM(model_name=model_key)


def _trunc(text: str, n: int = MAX_DISPLAY_LEN) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + "…"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    metric_fn: Callable = get_metric_fn_rouge_L()

    # Load BM25 retrieval (shared key: use flanT5Small as canonical source)
    retrieval_fp = ROOT / "retrieval" / "retrieval_results" / "flanT5Small" / "bm25" / f"{LAMP_NUM}.json"
    with retrieval_fp.open() as f:
        retrieval_results: dict = json.load(f)

    # Gather first N_QUERIES qids from flanT5Small data
    # inputs JSON is a plain list; outputs JSON is {"task": ..., "golds": [...]}
    data_dir = ROOT / "data" / "lamp_utility_labels_flanT5Small"
    with (data_dir / f"{LAMP_NUM}_user_dev_inputs.json").open() as f:
        all_inputs: list[dict] = json.load(f)   # list of dicts
    with (data_dir / f"{LAMP_NUM}_user_dev_outputs.json").open() as f:
        raw_outputs = json.load(f)
    all_outputs: list[dict] = raw_outputs["golds"]  # list of {"id": ..., "output": ...}

    inputs_10 = all_inputs[:N_QUERIES]
    outputs_10 = all_outputs[:N_QUERIES]

    # Build table rows: one entry per query, columns per model
    rows: list[dict] = []
    for inp, out in zip(inputs_10, outputs_10):
        assert inp["id"] == out["id"]
        rows.append({
            "qid": inp["id"],
            "query": inp["input"],
            "gold": out["output"],
        })

    # Collect model predictions column by column
    for model_key in MODELS:
        print(f"\n{'='*60}", flush=True)
        print(f"  Running model: {model_key}", flush=True)
        print(f"{'='*60}", flush=True)

        tok_source = _resolve_tokenizer_source(model_key)
        tokenizer = AutoTokenizer.from_pretrained(tok_source, local_files_only=False)
        tok_max = max(1, int(tokenizer.model_max_length * 0.8))

        lamp_handler = LaMPHandler(
            lamp_dir_name=f"lamp_utility_labels_{model_key}",
            split_type="user",
            tokenizer_model_name=tok_source,
            k=K,
        )
        aip_func = lamp_handler.get_aip_func(lamp_num=LAMP_NUM)
        generator = _build_generator(model_key)

        preds: list[str] = []
        for inp, out in zip(inputs_10, outputs_10):
            qid = inp["id"]
            question = inp["input"]
            rr = retrieval_results.get(qid, [])
            scores = np.array([float(x[1]) for x in rr], dtype=float)

            # deterministic top-k by score
            top_indices = np.argsort(-scores, kind="mergesort")[:K]
            pids = [rr[i][0] for i in top_indices]
            selected_profiles = lamp_handler.find_profiles_by_pids(LAMP_NUM, qid, pids)

            prompt = aip_func(question=question, profiles=selected_profiles)
            prompt = trim_sentence_by_token_len(prompt, tokenizer=tokenizer, max_tok_len=tok_max)

            pred = generator.answer_question(final_prompt=prompt).strip()
            if pred == "" or all(c in {".", " "} for c in pred):
                pred = "<empty>"
            preds.append(pred)
            print(f"  [{qid}] pred: {_trunc(pred)}", flush=True)

        scores_list = metric_fn(preds, [r["gold"] for r in rows])
        for i, (pred, score) in enumerate(zip(preds, scores_list)):
            rows[i][f"pred_{model_key}"] = pred
            rows[i][f"rl_{model_key}"] = round(score, 4)

        # Free model memory between runs
        del generator

    # ── Print results table ───────────────────────────────────────────────────
    print("\n\n" + "="*80)
    print("QUERY-LEVEL COMPARISON  (LaMP-4, BM25 top-5, deterministic)")
    print("="*80)

    for i, row in enumerate(rows):
        print(f"\n{'─'*80}")
        print(f"Query {i+1:2d}  qid={row['qid']}")
        print(f"  Input  : {_trunc(row['query'], 200)}")
        print(f"  Gold   : {_trunc(row['gold'],  200)}")
        for model_key in MODELS:
            score = row.get(f"rl_{model_key}", float("nan"))
            pred  = row.get(f"pred_{model_key}", "–")
            print(f"  [{model_key:20s}] RL={score:.4f}  →  {_trunc(pred, 160)}")

    print(f"\n{'─'*80}")
    print("AVERAGES:")
    for model_key in MODELS:
        all_scores = [row[f"rl_{model_key}"] for row in rows if f"rl_{model_key}" in row]
        avg = sum(all_scores) / len(all_scores) if all_scores else float("nan")
        print(f"  {model_key:20s}  mean rouge-L = {avg:.4f}")

    # Also dump as a DataFrame for easy inspection
    df = pd.DataFrame(rows)
    print("\n\nFULL COMPARISON TABLE (CSV-friendly):")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
