"""Quick check that MLX output cleaning works for LFM and Qwen."""
import json, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import models_info, trim_sentence_by_token_len
from data.lamp_handler import LaMPHandler
from eval.lamp_metrics import get_metric_fn_rouge_L
from generator.lm_mlx import PromptLMMLX
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

LAMP_NUM, N, K = 4, 10, 5
MODELS = ["lfm25MLX12B4bit", "qwen35MLX4B4bit"]

root = Path(".")
rr = json.load((root / "retrieval/retrieval_results/flanT5Small/bm25/4.json").open())
inputs = json.load((root / "data/lamp_utility_labels_flanT5Small/4_user_dev_inputs.json").open())[:N]
outputs = json.load((root / "data/lamp_utility_labels_flanT5Small/4_user_dev_outputs.json").open())["golds"][:N]

rows = [{"qid": a["id"], "gold": b["output"]} for a, b in zip(inputs, outputs)]
metric = get_metric_fn_rouge_L()

def resolve_tok(key):
    mid = models_info[key]["model_id"]
    try:
        return str(snapshot_download(mid, local_files_only=True))
    except Exception:
        return str(snapshot_download(mid, local_files_only=False))

for mkey in MODELS:
    print(f"\n====== {mkey} ======", flush=True)
    tok_src = resolve_tok(mkey)
    tok = AutoTokenizer.from_pretrained(tok_src, local_files_only=False)
    handler = LaMPHandler(
        lamp_dir_name=f"lamp_utility_labels_{mkey}",
        split_type="user",
        tokenizer_model_name=tok_src,
        k=K,
    )
    aip = handler.get_aip_func(lamp_num=LAMP_NUM)
    gen = PromptLMMLX(model_name=mkey, model_kwargs=models_info[mkey].get("model_kwargs", {}))
    preds = []
    for inp in inputs:
        rr_q = rr.get(inp["id"], [])
        sc = np.array([float(x[1]) for x in rr_q])
        top = np.argsort(-sc)[:K]
        pids = [rr_q[i][0] for i in top]
        profs = handler.find_profiles_by_pids(LAMP_NUM, inp["id"], pids)
        prompt = aip(question=inp["input"], profiles=profs)
        prompt = trim_sentence_by_token_len(
            prompt, tokenizer=tok, max_tok_len=max(1, int(tok.model_max_length * 0.8))
        )
        pred = gen.answer_question(final_prompt=prompt).strip() or "<empty>"
        preds.append(pred)
        print(f"  [{inp['id']}] {pred[:140]}", flush=True)
    scores = metric(preds, [r["gold"] for r in rows])
    print(f"  mean RL = {sum(scores)/len(scores):.4f}  (10 queries)", flush=True)
    del gen
