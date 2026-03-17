"""Quick smoke-test: validates PL sampling + EE pipeline without LLM."""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from perturbation import plackettluce as pl
from utils import make_trec_top_file_for_single_qid, make_trec_rel_file_for_single_qid
from expected_exposure import expeval

print("Loading BM25 retrieval results...")
with open("retrieval/retrieval_results/flanT5Small/bm25/4.json") as f:
    retrieval_results = json.load(f)
print(f"  {len(retrieval_results)} queries loaded")

ALPHA = 2
K = 5
N_SAMPLES = 10
REL_MAP = "data/lamp_utility_labels_flanT5Small/4_relevance_mapping.tsv"

ee_params = {
    "umType": "rbp", "umPatience": 1, "umUtility": 0.5,
    "binarize": False, "groupEvaluation": False, "complete": False,
    "normalize": True, "relfn": "", "topfn": "",
}

qids = list(retrieval_results.keys())[:2]
print(f"Testing {len(qids)} queries ...")
for qid in qids:
    scores = np.array([p[1] for p in retrieval_results[qid]])
    mn, mx = scores.min(), scores.max()
    if mn < 0:
        scores -= mn
    if mx != mn:
        scores = (scores - mn) / (mx - mn)
    scores = scores + 1
    scores = scores ** ALPHA
    pl_result = pl.gumbel_sample_rankings(scores, N_SAMPLES, cutoff=K, doc_prob=False)
    sampled = pl_result[0]
    top_fp = make_trec_top_file_for_single_qid(qid, sampled, retrieval_results[qid], "test")
    rel_fp = make_trec_rel_file_for_single_qid(qid, REL_MAP)
    ee_params["topfn"] = top_fp
    ee_params["relfn"] = rel_fp
    result = expeval.run(parameters=ee_params, k=K)
    os.remove(top_fp)
    os.remove(rel_fp)
    print(f"  qid={qid}  EE-D={result['disparity']:.4f}  EE-R={result['relevance']:.4f}")

print("EE smoke test PASSED")
