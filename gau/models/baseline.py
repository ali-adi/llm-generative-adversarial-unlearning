import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class BaselineModel:
    def __init__(self, model_name, tokenizer_name, device="cuda", use_fp16=True, max_seq_length=512):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if use_fp16:
            self.model = self.model.half()
        self.model.to(self.device)
        self.max_seq_length = max_seq_length

    def generate_response(
        self, prompt, 
        max_new_tokens=128, 
        temperature=0.7, 
        top_p=0.9, 
        do_sample=True
    ):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_seq_length).to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # Remove the prompt from the response if present
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        return response
