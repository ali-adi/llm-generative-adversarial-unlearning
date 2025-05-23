import json
import random
from typing import List, Dict, Set
import logging
import os

def load_jsonl(file_path: str) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def extract_questions(data: List[Dict], question_field: str = "question") -> List[str]:
    """Extract questions from a list of dicts using the specified field."""
    questions = []
    for item in data:
        if question_field in item:
            questions.append(item[question_field])
        elif "text" in item and item["text"].endswith("?"):
            questions.append(item["text"])
    return questions

def filter_overlap(questions: List[str], overlap_set: Set[str]) -> List[str]:
    """Remove questions that overlap with a given set."""
    return [q for q in questions if q not in overlap_set]

def sample_questions(questions: List[str], num: int, seed: int = 42) -> List[str]:
    """Randomly sample a specified number of questions."""
    random.seed(seed)
    if len(questions) <= num:
        return questions
    return random.sample(questions, num)

def save_questions_jsonl(questions: List[str], out_path: str):
    """Save questions as JSONL with 'query' key."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps({"query": q}) + "\n")
    logging.info(f"Saved {len(questions)} questions to {out_path}")

def process_and_save_questions(
    raw_file: str,
    out_file: str,
    num_questions: int,
    question_field: str = "question",
    overlap_set: Set[str] = None,
    seed: int = 42,
):
    """Load, extract, filter, sample, and save questions."""
    data = load_jsonl(raw_file)
    questions = extract_questions(data, question_field)
    if overlap_set:
        questions = filter_overlap(questions, overlap_set)
    questions = sample_questions(questions, num_questions, seed)
    save_questions_jsonl(questions, out_file)
    return questions
