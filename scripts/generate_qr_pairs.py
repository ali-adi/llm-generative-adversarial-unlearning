import argparse
import yaml
import json
import logging
from tqdm import tqdm
from gau.models.baseline import BaselineModel

def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line)["query"] for line in f]

def save_qr_pairs(qr_pairs, out_path, metadata=None):
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in qr_pairs:
            entry = dict(pair)
            if metadata:
                entry["metadata"] = metadata
            f.write(json.dumps(entry) + "\n")

def main(data_config_path, model_config_path):
    # Load configs
    with open(data_config_path, "r") as f:
        data_cfg = yaml.safe_load(f)
    with open(model_config_path, "r") as f:
        model_cfg = yaml.safe_load(f)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s"
    )

    # Load model
    logging.info("Loading baseline model...")
    model = BaselineModel(
        model_name=model_cfg["model_name"],
        tokenizer_name=model_cfg["tokenizer_name"],
        device=model_cfg.get("device", "cuda"),
        use_fp16=model_cfg.get("use_fp16", True),
        max_seq_length=model_cfg.get("max_seq_length", 512)
    )

    # Generate for both target and retained
    for kind in ["target", "retained"]:
        question_file = data_cfg[f"{kind}_qr_file"].replace(".jsonl", "_questions.jsonl")
        out_file = data_cfg[f"{kind}_qr_file"]
        logging.info(f"Generating Q-R pairs for {kind} knowledge...")
        questions = load_questions(question_file)
        qr_pairs = []
        for q in tqdm(questions):
            response = model.generate_response(q)
            qr_pairs.append({"query": q, "response": response})
        metadata = {
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "model_version": model_cfg["model_name"]
        }
        save_qr_pairs(qr_pairs, out_file, metadata=metadata)
        logging.info(f"Saved {len(qr_pairs)} Q-R pairs to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", type=str, default="configs/data_config.yaml")
    parser.add_argument("--model_config", type=str, default="configs/model_config.yaml")
    args = parser.parse_args()
    main(args.data_config, args.model_config)
