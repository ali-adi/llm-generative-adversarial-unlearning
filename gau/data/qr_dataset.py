import json
import torch
from torch.utils.data import Dataset

class QRDataset(Dataset):
    """
    Dataset for Q-R pairs.
    Each item is a dict: {"query": ..., "response": ...}
    Supports:
      - mode="generator": for language modeling (input: query, label: response)
      - mode="discriminator": for classification (input: query + response, label: optional)
    """
    def __init__(
        self,
        file_path,
        tokenizer,
        max_seq_length=512,
        mode="generator",
        labels=None,  # Optional: list/array of labels for discriminator mode
        sep_token=None
    ):
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mode = mode
        self.labels = labels
        # Use tokenizer.sep_token if available, else fallback to '[SEP]'
        self.sep_token = sep_token or getattr(tokenizer, "sep_token", "[SEP]")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item["query"]
        response = item.get("response", "")

        if self.mode == "generator":
            # For generator: input is query, target is response
            inputs = self.tokenizer(
                query,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            )
            targets = self.tokenizer(
                response,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            )
            return {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "labels": targets["input_ids"].squeeze(0),
                "query": query,
                "response": response
            }
        elif self.mode == "discriminator":
            # For discriminator: input is query + [SEP] + response
            combined = f"{query} {self.sep_token} {response}"
            inputs = self.tokenizer(
                combined,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            )
            out = {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "query": query,
                "response": response
            }
            # Optionally add label if provided
            if self.labels is not None:
                out["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return out
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

