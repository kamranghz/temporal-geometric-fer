import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional

from .confusion_utils import build_soft_labels, update_confusion_matrix, row_normalize


class EMAStabilityTrainer:
    """
    Training wrapper that uses an EMA confusion matrix to construct per-sample
    soft labels, reducing overconfidence and neutral-class bias.

    Inspired by MER-SoLa (Yun et al., 2025).  At each epoch the confusion
    matrix observed on the training set is blended into a running EMA estimate.
    That EMA is used to build "confusion-aware" soft labels on the next epoch,
    so that classes frequently confused with the true label receive non-zero
    supervision signal.

    Loss::

        L = λ · CE_weighted(logits, y_hard)
          + (1 - λ) · NLL(logits, ỹ_soft)

    Args:
        model:         Emotion recognition model returning logits of shape (B, K).
        num_classes:   Number of emotion classes (default 8 for AffectNet-8).
        beta:          EMA decay for the confusion matrix (high → slow update).
        delta:         Confusion threshold for soft-label construction.
        lambda_weight: Balance between hard CE and soft NLL loss (λ=1 → pure CE).
        temperature:   Softmax temperature applied when building soft labels.
        class_weights: Optional tensor of shape (K,) for class-imbalance weighting.
        device:        Torch device string ('cuda' or 'cpu').
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 8,
        beta: float = 0.9,
        delta: float = 0.15,
        lambda_weight: float = 0.8,
        temperature: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.num_classes = num_classes
        self.beta = beta
        self.delta = delta
        self.lambda_weight = lambda_weight
        self.temperature = temperature
        self.device = device

        self.cm_ema = torch.eye(num_classes, device=device)
        self.class_weights = (
            class_weights.to(device)
            if class_weights is not None
            else torch.ones(num_classes, device=device)
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self, logits: torch.Tensor, y_true: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the hybrid hard/soft loss for a batch.

        Args:
            logits: Raw model output, shape (B, K).
            y_true: Integer ground-truth labels, shape (B,).

        Returns:
            Tuple of (total_loss, ce_loss, soft_nll_loss).
        """
        log_probs = F.log_softmax(logits, dim=1)

        ce_loss = F.nll_loss(log_probs, y_true, weight=self.class_weights)

        soft_labels = torch.stack(
            [build_soft_labels(self.cm_ema[y.item()], y.item(), self.delta, self.temperature)
             for y in y_true]
        )
        soft_nll = -(soft_labels * log_probs).sum(dim=1).mean()

        total = self.lambda_weight * ce_loss + (1.0 - self.lambda_weight) * soft_nll
        return total, ce_loss, soft_nll

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_epoch(
        self, loader: DataLoader, optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Train for one epoch; returns loss and accuracy."""
        self.model.train()
        cm_counts = torch.zeros(self.num_classes, self.num_classes, device=self.device)
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            logits = self.model(images)
            loss, _, _ = self.compute_loss(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                cm_counts = update_confusion_matrix(cm_counts, labels.cpu(), preds.cpu(), self.num_classes)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()

        # Update EMA confusion matrix after epoch
        cm_epoch = row_normalize(cm_counts)
        self.cm_ema = self.beta * self.cm_ema + (1.0 - self.beta) * cm_epoch

        return {"loss": total_loss / len(loader), "accuracy": 100.0 * correct / total}

    def validate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate on a validation set; returns accuracy."""
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                preds = self.model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return {"accuracy": 100.0 * correct / total}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        lr: float = 1e-3,
    ) -> float:
        """Full training loop. Returns best validation accuracy."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, optimizer)
            val_metrics = self.validate(val_loader)

            marker = " *" if val_metrics["accuracy"] > best_val_acc else ""
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"loss={train_metrics['loss']:.4f}  "
                f"train_acc={train_metrics['accuracy']:.1f}%  "
                f"val_acc={val_metrics['accuracy']:.1f}%{marker}"
            )

            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]

        return best_val_acc
