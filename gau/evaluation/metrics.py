import torch
import torch.nn.functional as F
from tqdm import tqdm

def compute_perplexity(model, dataloader, device="cuda"):
    """
    Computes perplexity of a model on a dataset.
    Assumes dataloader yields dicts with 'input_ids', 'attention_mask', 'labels'.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Perplexity"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    model.train()
    return torch.exp(torch.tensor(total_loss / total_tokens))

def compute_accuracy(model, dataloader, device="cuda"):
    """
    Computes accuracy for classification (discriminator).
    Assumes dataloader yields dicts with 'input_ids', 'attention_mask', 'labels'.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Accuracy"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            if logits.shape[-1] == 2:
                preds = torch.argmax(logits, dim=-1)
            else:
                preds = (torch.sigmoid(logits) > 0.5).long().squeeze()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    model.train()
    return correct / total if total > 0 else 0.0

def evaluate_qa(generator, qa_dataset, tokenizer, device="cuda"):
    correct = 0
    total = 0
    for item in qa_dataset:
        question = item["question"]
        gt_answer = item["answer"]
        pred = generator.generate(question)
        if pred.strip().lower() == gt_answer.strip().lower():
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0
