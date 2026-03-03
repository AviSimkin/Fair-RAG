import argparse
from pathlib import Path
import pandas as pd


def main(generator_name: str, lamp_nums: list[int]) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / f"lamp_utility_labels_{generator_name}"
    out_dir = repo_root / "utility_labels" / "eval_results" / generator_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for lamp_num in lamp_nums:
        mapping_fp = data_dir / f"{lamp_num}_relevance_mapping.tsv"
        if not mapping_fp.exists():
            raise FileNotFoundError(f"Missing relevance mapping: {mapping_fp}")

        mapping_df = pd.read_csv(mapping_fp, sep="\t", dtype={"qid": str, "pid": str})
        if "relevance_label" not in mapping_df.columns:
            raise ValueError(f"Missing 'relevance_label' column in {mapping_fp}")

        delta_df = mapping_df[["qid", "pid", "relevance_label"]].copy()
        delta_df["delta"] = delta_df["relevance_label"].astype(float)
        delta_df = delta_df[["qid", "pid", "delta"]]

        out_fp = out_dir / f"{lamp_num}_delta.tsv"
        delta_df.to_csv(out_fp, sep="\t", index=False)
        print(f"Wrote: {out_fp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap utility_labels/eval_results/*_delta.tsv from provided relevance mappings"
    )
    parser.add_argument("--generator_name", type=str, required=True)
    parser.add_argument("--lamp_nums", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    args = parser.parse_args()

    main(generator_name=args.generator_name, lamp_nums=args.lamp_nums)
