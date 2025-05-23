import yaml
import logging
import os
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from gau.data.qr_dataset import QRDataset
from gau.data.utils import get_dataloader
from gau.models.generator import GeneratorModel
from gau.models.discriminator import DiscriminatorModel
from gau.training.trainer import GAUTrainer
from gau.evaluation.metrics import compute_perplexity, compute_accuracy
from gau.evaluation.qualitative import sample_qr_pairs, print_qualitative
from gau.utils.resource import ResourceMonitor
from gau.utils.checkpoint import save_checkpoint, load_checkpoint
from gau.training.early_stopping import EarlyStopping
from torch.profiler import profile, record_function, ProfilerActivity
from torch.optim import AdamW

def setup_logging(log_dir="logs/"):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "main.log")),
            logging.StreamHandler()
        ]
    )

def main():
    # === Load configs ===
    with open("configs/model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open("configs/training_config.yaml") as f:
        train_cfg = yaml.safe_load(f)
    with open("configs/data_config.yaml") as f:
        data_cfg = yaml.safe_load(f)
    with open("configs/eval_config.yaml") as f:
        eval_cfg = yaml.safe_load(f)

    setup_logging()

    device = model_cfg.get("device", "cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["tokenizer_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === Prepare DataLoaders ===
    target_dataset = QRDataset(
        file_path=data_cfg["target_qr_file"],
        tokenizer=tokenizer,
        max_seq_length=model_cfg.get("max_seq_length", 512),
        mode="discriminator"
    )
    retained_dataset = QRDataset(
        file_path=data_cfg["retained_qr_file"],
        tokenizer=tokenizer,
        max_seq_length=model_cfg.get("max_seq_length", 512),
        mode="discriminator"
    )
    target_loader = get_dataloader(
        target_dataset,
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True),
        tokenizer=tokenizer,
        max_seq_length=model_cfg.get("max_seq_length", 512),
        mode="discriminator"
    )
    retained_loader = get_dataloader(
        retained_dataset,
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True),
        tokenizer=tokenizer,
        max_seq_length=model_cfg.get("max_seq_length", 512),
        mode="discriminator"
    )

    # === Initialize Models ===
    generator = GeneratorModel(
        model_name=model_cfg["model_name"],
        tokenizer_name=model_cfg["tokenizer_name"],
        device=device,
        use_fp16=model_cfg.get("use_fp16", True),
        max_seq_length=model_cfg.get("max_seq_length", 512)
    )
    discriminator = DiscriminatorModel(
        model_name=model_cfg["model_name"],
        tokenizer_name=model_cfg["tokenizer_name"],
        device=device,
        use_fp16=model_cfg.get("use_fp16", True),
        max_seq_length=model_cfg.get("max_seq_length", 512),
        discriminator_type=model_cfg.get("discriminator_type", "prompt"),
        num_labels=2
    )

    # === Optimizers ===
    gen_optimizer = AdamW(generator.model.parameters(), lr=float(train_cfg["learning_rate"]))
    disc_optimizer = AdamW(discriminator.parameters(), lr=float(train_cfg["learning_rate"]))

    # === Schedulers ===
    steps_per_epoch = len(target_loader)
    num_epochs = train_cfg.get("num_epochs", 3)
    total_steps = steps_per_epoch * num_epochs
    gen_scheduler = get_linear_schedule_with_warmup(
        gen_optimizer,
        num_warmup_steps=train_cfg.get("lr_scheduler_warmup_steps", 100),
        num_training_steps=total_steps
    )
    disc_scheduler = get_linear_schedule_with_warmup(
        disc_optimizer,
        num_warmup_steps=train_cfg.get("lr_scheduler_warmup_steps", 100),
        num_training_steps=total_steps
    )

    # === Resource Monitor ===
    resource_monitor = ResourceMonitor(log_file="logs/resource_log.txt")

    # === Checkpoint Resume (optional) ===
    start_epoch = 0
    if model_cfg.get("resume_from_checkpoint"):
        start_epoch = load_checkpoint(
            generator, discriminator, gen_optimizer, disc_optimizer,
            checkpoint_path=model_cfg["resume_from_checkpoint"],
            device=device
        )
        logging.info(f"Resumed from checkpoint at epoch {start_epoch}")

    # === Early Stopping ===
    patience = train_cfg.get("early_stopping_patience", 3)
    min_delta = train_cfg.get("early_stopping_min_delta", 0.0)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    # === Trainer ===
    train_loaders = {"target": target_loader, "retained": retained_loader}
    trainer = GAUTrainer(
        generator=generator,
        discriminator=discriminator,
        gen_optimizer=gen_optimizer,
        disc_optimizer=disc_optimizer,
        train_loaders=train_loaders,
        tokenizer=tokenizer,
        device=device,
        config=train_cfg,
        resource_monitor=resource_monitor,
        gen_scheduler=gen_scheduler,
        disc_scheduler=disc_scheduler
    )

    # === Training ===
    num_epochs = train_cfg.get("num_epochs", 3)
    for epoch in range(start_epoch, num_epochs):
        if epoch == 0:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                          schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                          on_trace_ready=torch.profiler.tensorboard_trace_handler('logs/torch_profiler'),
                          record_shapes=True,
                          profile_memory=True,
                          with_stack=True) as prof:
                trainer.train_one_epoch(epoch)
                prof.step()
        else:
            trainer.train_one_epoch(epoch)
        # Save checkpoint
        save_checkpoint(
            generator, discriminator, gen_optimizer, disc_optimizer,
            epoch, model_cfg.get("checkpoint_dir", "checkpoints/")
        )
        # Evaluate after each epoch
        for kind in ["target", "retained"]:
            eval_dataset = QRDataset(
                file_path=data_cfg[f"{kind}_qr_file"],
                tokenizer=tokenizer,
                max_seq_length=model_cfg.get("max_seq_length", 512),
                mode="generator"
            )
            eval_loader = get_dataloader(
                eval_dataset,
                batch_size=train_cfg.get("batch_size", 8),
                shuffle=False,
                num_workers=train_cfg.get("num_workers", 4),
                pin_memory=train_cfg.get("pin_memory", True),
                tokenizer=tokenizer,
                max_seq_length=model_cfg.get("max_seq_length", 512),
                mode="generator"
            )
            ppl = compute_perplexity(generator.model, eval_loader, device=device)
            logging.info(f"Perplexity on {kind} knowledge after epoch {epoch+1}: {ppl.item():.4f}")
            if kind == "retained":
                retained_ppl = ppl.item()
        # Track discriminator accuracy on target knowledge for early stopping
        eval_dataset = QRDataset(
            file_path=data_cfg["target_qr_file"],
            tokenizer=tokenizer,
            max_seq_length=model_cfg.get("max_seq_length", 512),
            mode="discriminator"
        )
        eval_loader = get_dataloader(
            eval_dataset,
            batch_size=train_cfg.get("batch_size", 8),
            shuffle=False,
            num_workers=train_cfg.get("num_workers", 4),
            pin_memory=train_cfg.get("pin_memory", True),
            tokenizer=tokenizer,
            max_seq_length=model_cfg.get("max_seq_length", 512),
            mode="discriminator"
        )
        target_acc = compute_accuracy(discriminator, eval_loader, device=device)
        logging.info(f"Discriminator accuracy on target knowledge after epoch {epoch+1}: {target_acc:.4f}")
        # Also log retained accuracy as before
        eval_dataset = QRDataset(
            file_path=data_cfg["retained_qr_file"],
            tokenizer=tokenizer,
            max_seq_length=model_cfg.get("max_seq_length", 512),
            mode="discriminator"
        )
        eval_loader = get_dataloader(
            eval_dataset,
            batch_size=train_cfg.get("batch_size", 8),
            shuffle=False,
            num_workers=train_cfg.get("num_workers", 4),
            pin_memory=train_cfg.get("pin_memory", True),
            tokenizer=tokenizer,
            max_seq_length=model_cfg.get("max_seq_length", 512),
            mode="discriminator"
        )
        retained_acc = compute_accuracy(discriminator, eval_loader, device=device)
        logging.info(f"Discriminator accuracy on retained knowledge after epoch {epoch+1}: {retained_acc:.4f}")
        resource_monitor.log_gpu_memory(step=f"epoch_{epoch+1}")
        resource_monitor.log_cpu_memory_disk()
        resource_monitor.log_gpu_utilization()
        # Early stopping check (using target accuracy)
        if early_stopping.step(target_acc):
            logging.info(f"Early stopping triggered at epoch {epoch+1} (patience={patience}, min_delta={min_delta})")
            break
        # Success thresholds enforcement
        thresholds = eval_cfg.get("success_thresholds", {})
        unlearned_acc_thresh = thresholds.get("unlearned_acc")
        retained_ppl_thresh = thresholds.get("retained_ppl")
        stop_for_success = False
        if unlearned_acc_thresh is not None and target_acc >= unlearned_acc_thresh:
            logging.info(f"Success threshold met: target accuracy {target_acc:.4f} >= {unlearned_acc_thresh}")
            stop_for_success = True
        if retained_ppl_thresh is not None and retained_ppl <= retained_ppl_thresh:
            logging.info(f"Success threshold met: retained PPL {retained_ppl:.4f} <= {retained_ppl_thresh}")
            stop_for_success = True
        if stop_for_success:
            logging.info(f"Stopping training due to success thresholds at epoch {epoch+1}")
            break

    resource_monitor.summary()
    resource_monitor.log_cpu_memory_disk()
    resource_monitor.log_gpu_utilization()

    # === Qualitative Review ===
    from gau.evaluation.qualitative import sample_qr_pairs, print_qualitative
    for kind in ["target", "retained"]:
        logging.info(f"\nQualitative review for {kind} knowledge:")
        samples = sample_qr_pairs(data_cfg[f"{kind}_qr_file"], num_samples=5)
        print_qualitative(generator, samples)

if __name__ == "__main__":
    main()
