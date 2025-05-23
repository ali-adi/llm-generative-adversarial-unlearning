import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch.nn as nn

class DiscriminatorModel(nn.Module):
    """
    Discriminator (Evaluation Model) for classifying Q-R pairs as 'Retained' or 'Unlearned'.
    Supports prompt-based or head-based classification.
    """
    def __init__(
        self,
        model_name,
        tokenizer_name,
        device="cuda",
        use_fp16=True,
        max_seq_length=512,
        discriminator_type="prompt",  # or "head"
        num_labels=2,
        train_top_n_layers=None
    ):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.discriminator_type = discriminator_type
        self.max_seq_length = max_seq_length

        if discriminator_type == "prompt":
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            if use_fp16:
                self.model = self.model.half()
            self.model.to(self.device)
        elif discriminator_type == "head":
            self.backbone = AutoModel.from_pretrained(model_name)
            if use_fp16:
                self.backbone = self.backbone.half()
            self.backbone.to(self.device)
            self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels)
            self.classifier.to(self.device)
        else:
            raise ValueError("discriminator_type must be 'prompt' or 'head'")

        # Optionally freeze all but top N layers
        if train_top_n_layers is not None and discriminator_type == "head":
            for name, param in self.backbone.named_parameters():
                if not any([f"layer.{i}." in name for i in range(self.backbone.config.num_hidden_layers - train_top_n_layers, self.backbone.config.num_hidden_layers)]):
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        if self.discriminator_type == "prompt":
            # Prompt-based: Use LLM to output a probability/logit for each class
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Use the logits of the last token as the class score (simplest approach)
            logits = outputs.logits[:, -1, :]  # (batch, vocab_size)
            # You may want to map logits to a binary class (e.g., via prompt engineering)
            return logits
        elif self.discriminator_type == "head":
            # Head-based: Use backbone + classifier
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            # Use the hidden state of the [CLS] token (first token)
            pooled = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)
            logits = self.classifier(pooled)  # (batch, num_labels)
            return logits
