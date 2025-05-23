import yaml
import torch
from transformers import AutoTokenizer
from gau.models.generator import GeneratorModel
from gau.models.discriminator import DiscriminatorModel
from gau.data.qr_dataset import QRDataset
from gau.data.utils import get_dataloader
from gau.evaluation.metrics import compute_perplexity, compute_accuracy

def main():
    # Load configs
    with open("configs/model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open("configs/data_config.yaml") as f:
        data_cfg = yaml.safe_load(f)
    with open("configs/training_config.yaml") as f:
        train_cfg = yaml.safe_load(f)

    device = model_cfg.get("device", "cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["tokenizer_name"])

    # Generator evaluation (perplexity on target and retained)
    generator = GeneratorModel(
        model_name=model_cfg["model_name"],
        tokenizer_name=model_cfg["tokenizer_name"],
        device=device,
        use_fp16=model_cfg.get("use_fp16", True),
        max_seq_length=model_cfg.get("max_seq_length", 512)
    )

    for kind in ["target", "retained"]:
        dataset = QRDataset(
            file_path=data_cfg[f"{kind}_qr_file"],
            tokenizer=tokenizer,
            max_seq_length=model_cfg.get("max_seq_length", 512),
            mode="generator"
        )
        dataloader = get_dataloader(
            dataset,
            batch_size=train_cfg.get("batch_size", 8),
            shuffle=False,
            num_workers=train_cfg.get("num_workers", 4),
            pin_memory=train_cfg.get("pin_memory", True)
        )
        ppl = compute_perplexity(generator.model, dataloader, device=device)
        print(f"Perplexity on {kind} knowledge: {ppl.item():.4f}")

    # Discriminator evaluation (accuracy)
    discriminator = DiscriminatorModel(
        model_name=model_cfg["model_name"],
        tokenizer_name=model_cfg["tokenizer_name"],
        device=device,
        use_fp16=model_cfg.get("use_fp16", True),
        max_seq_length=model_cfg.get("max_seq_length", 512),
        discriminator_type=model_cfg.get("discriminator_type", "prompt"),
        num_labels=2
    )

    for kind, label in [("target", 1), ("retained", 0)]:
        dataset = QRDataset(
            file_path=data_cfg[f"{kind}_qr_file"],
            tokenizer=tokenizer,
            max_seq_length=model_cfg.get("max_seq_length", 512),
            mode="discriminator"
        )
        # Add labels to dataset items
        def add_label(batch):
            batch["labels"] = torch.full((len(batch["input_ids"]),), label, dtype=torch.long)
            return batch

        dataloader = get_dataloader(
            dataset,
            batch_size=train_cfg.get("batch_size", 8),
            shuffle=False,
            num_workers=train_cfg.get("num_workers", 4),
            pin_memory=train_cfg.get("pin_memory", True),
            collate_fn=None  # You can add a custom collate_fn if needed
        )
        acc = compute_accuracy(discriminator, dataloader, device=device)
        print(f"Discriminator accuracy on {kind} knowledge: {acc:.4f}")

if __name__ == "__main__":
    main()
