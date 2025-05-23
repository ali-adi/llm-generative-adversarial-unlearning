import torch
import torch.nn.functional as F

def discriminator_loss(logits, labels, label_smoothing=0.0):
    """
    Binary cross-entropy loss for discriminator.
    logits: (batch, 2) or (batch,) if using a single logit
    labels: (batch,) with values 0 (retained) or 1 (unlearned)
    """
    if logits.shape[-1] == 2:
        # If logits are (batch, 2), use CrossEntropyLoss
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        return loss_fn(logits, labels)
    else:
        # If logits are (batch,), use BCEWithLogitsLoss
        loss_fn = torch.nn.BCEWithLogitsLoss()
        return loss_fn(logits.squeeze(), labels.float())

def generator_loss(disc_logits, target_label=1):
    """
    Generator tries to fool the discriminator into predicting 'unlearned' (label=1).
    disc_logits: Discriminator's output on generator's Q-R pairs.
    """
    # If logits are (batch, 2), use CrossEntropyLoss with all labels=1
    if disc_logits.shape[-1] == 2:
        labels = torch.ones(disc_logits.size(0), dtype=torch.long, device=disc_logits.device) * target_label
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(disc_logits, labels)
    else:
        # If logits are (batch,), use BCEWithLogitsLoss with all labels=1
        labels = torch.ones_like(disc_logits)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        return loss_fn(disc_logits.squeeze(), labels)
