import torch
import os

def save_checkpoint(generator, discriminator, gen_optimizer, disc_optimizer, epoch, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({
        "generator_state_dict": generator.model.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "gen_optimizer_state_dict": gen_optimizer.state_dict(),
        "disc_optimizer_state_dict": disc_optimizer.state_dict(),
        "epoch": epoch
    }, os.path.join(checkpoint_dir, f"gau_checkpoint_epoch{epoch}.pt"))

def load_checkpoint(generator, discriminator, gen_optimizer, disc_optimizer, checkpoint_path, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.model.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    gen_optimizer.load_state_dict(checkpoint["gen_optimizer_state_dict"])
    disc_optimizer.load_state_dict(checkpoint["disc_optimizer_state_dict"])
    return checkpoint.get("epoch", 0)
