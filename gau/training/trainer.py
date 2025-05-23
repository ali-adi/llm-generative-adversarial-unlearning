import torch
import logging
from tqdm import tqdm
from gau.training.losses import discriminator_loss, generator_loss

class GAUTrainer:
    def __init__(
        self,
        generator,
        discriminator,
        gen_optimizer,
        disc_optimizer,
        train_loaders,  # dict: {"target": DataLoader, "retained": DataLoader}
        tokenizer,
        device="cuda",
        config=None,
        resource_monitor=None,
        gen_scheduler=None,
        disc_scheduler=None
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.train_loaders = train_loaders
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or {}
        self.resource_monitor = resource_monitor
        self.gen_scheduler = gen_scheduler
        self.disc_scheduler = disc_scheduler
        self.gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        self.use_amp = self.config.get("use_amp", True)
        if self.use_amp:
            self.gen_scaler = torch.cuda.amp.GradScaler()
            self.disc_scaler = torch.cuda.amp.GradScaler()
        else:
            self.gen_scaler = None
            self.disc_scaler = None

    def train(self, num_epochs=1):
        for epoch in range(num_epochs):
            logging.info(f"Epoch {epoch+1}/{num_epochs}")
            self.train_one_epoch(epoch)
            if self.resource_monitor:
                self.resource_monitor.log_gpu_memory(step=f"epoch_{epoch+1}")

    def train_one_epoch(self, epoch):
        target_loader = self.train_loaders["target"]
        retained_loader = self.train_loaders["retained"]
        retained_iter = iter(retained_loader)

        disc_steps_per_gen = self.config.get("discriminator_steps_per_generator_step", 1)
        target_iter = iter(target_loader)
        num_batches = len(target_loader)
        i = 0
        step = 0
        while i < num_batches:
            # === Discriminator update(s) ===
            for _ in range(disc_steps_per_gen):
                try:
                    target_batch = next(target_iter)
                except StopIteration:
                    break
                # Only zero grad every accumulation cycle
                if step % self.gradient_accumulation_steps == 0:
                    self.disc_optimizer.zero_grad()
                try:
                    retained_batch = next(retained_iter)
                except StopIteration:
                    retained_iter = iter(retained_loader)
                    retained_batch = next(retained_iter)
                retained_inputs = {k: v.to(self.device) for k, v in retained_batch.items() if k in ["input_ids", "attention_mask"]}
                target_inputs = {k: v.to(self.device) for k, v in target_batch.items() if k in ["input_ids", "attention_mask"]}
                retained_labels = torch.zeros(retained_inputs["input_ids"].size(0), dtype=torch.long, device=self.device)
                target_labels = torch.ones(target_inputs["input_ids"].size(0), dtype=torch.long, device=self.device)
                # AMP forward/loss
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        retained_logits = self.discriminator(**retained_inputs)
                        target_logits = self.discriminator(**target_inputs)
                        logits = torch.cat([retained_logits, target_logits], dim=0)
                        labels = torch.cat([retained_labels, target_labels], dim=0)
                        disc_loss = discriminator_loss(logits, labels, label_smoothing=self.config.get("label_smoothing", 0.0))
                        disc_loss = disc_loss / self.gradient_accumulation_steps
                    self.disc_scaler.scale(disc_loss).backward()
                else:
                    retained_logits = self.discriminator(**retained_inputs)
                    target_logits = self.discriminator(**target_inputs)
                    logits = torch.cat([retained_logits, target_logits], dim=0)
                    labels = torch.cat([retained_labels, target_labels], dim=0)
                    disc_loss = discriminator_loss(logits, labels, label_smoothing=self.config.get("label_smoothing", 0.0))
                    disc_loss = disc_loss / self.gradient_accumulation_steps
                    disc_loss.backward()
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.get("max_grad_norm", 1.0))
                    if self.use_amp:
                        self.disc_scaler.step(self.disc_optimizer)
                        self.disc_scaler.update()
                    else:
                        self.disc_optimizer.step()
                    if self.disc_scheduler:
                        self.disc_scheduler.step()
                    if i % 10 == 0:
                        logging.info(f"Step {i}: disc_loss={disc_loss.item():.4f}")
                        if self.resource_monitor:
                            self.resource_monitor.log_gpu_memory(step=f"step_{i}")
                i += 1
                step += 1
            # === Generator update ===
            if step % self.gradient_accumulation_steps == 0:
                self.gen_optimizer.zero_grad()
            queries = target_batch["query"]  # List of strings
            generated_responses = []
            for q in queries:
                response = self.generator.generate(q)
                generated_responses.append(response)
            combined_texts = [f"{q} [SEP] {r}" for q, r in zip(queries, generated_responses)]
            gen_inputs = self.tokenizer(
                combined_texts,
                truncation=True,
                max_length=self.generator.max_seq_length,
                padding=True,
                return_tensors="pt"
            )
            gen_inputs = {k: v.to(self.device) for k, v in gen_inputs.items()}
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    gen_logits = self.discriminator(**gen_inputs)
                    gen_loss = generator_loss(gen_logits, target_label=1)
                    gen_loss = gen_loss / self.gradient_accumulation_steps
                self.gen_scaler.scale(gen_loss).backward()
            else:
                gen_logits = self.discriminator(**gen_inputs)
                gen_loss = generator_loss(gen_logits, target_label=1)
                gen_loss = gen_loss / self.gradient_accumulation_steps
                gen_loss.backward()
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.generator.model.parameters(), self.config.get("max_grad_norm", 1.0))
                if self.use_amp:
                    self.gen_scaler.step(self.gen_optimizer)
                    self.gen_scaler.update()
                else:
                    self.gen_optimizer.step()
                if self.gen_scheduler:
                    self.gen_scheduler.step()
                if i % 10 == 0:
                    logging.info(f"Step {i}: gen_loss={gen_loss.item():.4f}")
                    if self.resource_monitor:
                        self.resource_monitor.log_gpu_memory(step=f"step_{i}")
            step += 1
        logging.info(f"Epoch {epoch+1} finished.")
