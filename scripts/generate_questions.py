import argparse
import yaml
import logging
from gau.data.dataset import process_and_save_questions, load_jsonl, extract_questions

def main(config_path):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_cfg = config

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s"
    )

    # Process target questions
    logging.info("Processing target questions...")
    target_questions = process_and_save_questions(
        raw_file=data_cfg["raw_data_dir"] + "/target_raw.jsonl",
        out_file=data_cfg["target_qr_file"].replace(".jsonl", "_questions.jsonl"),
        num_questions=data_cfg["num_target_qr"],
        question_field="question",
        overlap_set=None,
        seed=data_cfg.get("random_seed", 42),
    )

    # Process retained questions
    logging.info("Processing retained questions...")
    # Optionally filter overlap with target questions
    overlap_set = set(target_questions) if data_cfg.get("filter_retained_overlap", False) else None
    process_and_save_questions(
        raw_file=data_cfg["raw_data_dir"] + "/retained_raw.jsonl",
        out_file=data_cfg["retained_qr_file"].replace(".jsonl", "_questions.jsonl"),
        num_questions=data_cfg["num_retained_qr"],
        question_field="question",
        overlap_set=overlap_set,
        seed=data_cfg.get("random_seed", 42),
    )

    logging.info("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data_config.yaml")
    args = parser.parse_args()
    main(args.config)
