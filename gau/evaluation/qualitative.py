import random
import json

def sample_qr_pairs(qr_file, num_samples=10):
    """Randomly sample Q-R pairs from a JSONL file."""
    with open(qr_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    samples = random.sample(lines, min(num_samples, len(lines)))
    return [json.loads(line) for line in samples]

def print_qualitative(generator, samples):
    """Print queries, ground truth, and generator outputs for manual review."""
    print("="*60)
    print("Qualitative Review of Generator Outputs")
    print("="*60)
    for i, item in enumerate(samples):
        query = item["query"]
        gt_response = item.get("response", "")
        gen_response = generator.generate(query)
        print(f"Sample {i+1}:")
        print(f"Query: {query}")
        print(f"Ground Truth Response: {gt_response}")
        print(f"Generator Output: {gen_response}")
        print("-"*60)
