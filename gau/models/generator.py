import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class GeneratorModel:
    """
    Generator (Unlearned Model) wrapper for LLM fine-tuning and generation.
    """
    def __init__(self, model_name, tokenizer_name, device="cuda", use_fp16=True, max_seq_length=512):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if use_fp16:
            self.model = self.model.half()
        self.model.to(self.device)
        self.max_seq_length = max_seq_length

    def generate(self, prompt, **gen_kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_seq_length).to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=gen_kwargs.get("max_new_tokens", 128),
                temperature=gen_kwargs.get("temperature", 0.7),
                top_p=gen_kwargs.get("top_p", 0.9),
                do_sample=gen_kwargs.get("do_sample", True),
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        return response

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
