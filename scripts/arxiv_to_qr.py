import os
import glob
import pandas as pd
import json

def parquet_to_qr(parquet_files, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for pq_file in parquet_files:
            df = pd.read_parquet(pq_file)
            for text in df["text"]:
                qr = {"query": text, "response": ""}
                f.write(json.dumps(qr) + "\n")
    print(f"Wrote {out_path} from {len(parquet_files)} parquet files.")

def main():
    base_in = "data/MU_RedPajama-Data-1T_1k_unlearn_1k_rest_reservoir_eng/arxiv"
    base_out = "data/processed/arxiv"
    os.makedirs(base_out, exist_ok=True)

    # Process forget (target)
    forget_files = sorted(glob.glob(os.path.join(base_in, "forget-*.parquet")))
    out_target = os.path.join(base_out, "target_qr.jsonl")
    parquet_to_qr(forget_files, out_target)

    # Process retain (retained)
    retain_files = sorted(glob.glob(os.path.join(base_in, "retain-*.parquet")))
    out_retained = os.path.join(base_out, "retained_qr.jsonl")
    parquet_to_qr(retain_files, out_retained)

    print("\nUpdate your configs/data_config.yaml as follows:")
    print(f"target_qr_file: {out_target}")
    print(f"retained_qr_file: {out_retained}")

if __name__ == "__main__":
    main() 