import torch
from torch.utils.data import DataLoader

def make_collate_fn(tokenizer, max_seq_length=512, mode="discriminator"):
    def collate(batch):
        # batch is a list of dicts with "query" and "response"
        queries = [item["query"] if (item["query"] is not None and isinstance(item["query"], str)) else "" for item in batch]
        responses = [item["response"] if (item["response"] is not None and isinstance(item["response"], str)) else "" for item in batch]
        if mode == "discriminator":
            # For discriminator, concatenate Q and R
            texts = [f"{q} [SEP] {r}" for q, r in zip(queries, responses)]
            encodings = tokenizer(
                texts,
                truncation=True,
                max_length=max_seq_length,
                padding=True,
                return_tensors="pt"
            )
            # If labels are present in the batch, add them
            if "label" in batch[0]:
                encodings["labels"] = torch.tensor([item["label"] for item in batch])
            return encodings
        else:
            # For generator, just tokenize queries (or both if needed)
            encodings = tokenizer(
                queries,
                truncation=True,
                max_length=max_seq_length,
                padding=True,
                return_tensors="pt"
            )
            if "label" in batch[0]:
                encodings["labels"] = torch.tensor([item["label"] for item in batch])
            return encodings
    return collate

def get_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory, tokenizer, max_seq_length=512, mode="discriminator"):
    collate_fn = make_collate_fn(tokenizer, max_seq_length, mode)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
