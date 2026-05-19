import torch
import torch.nn.functional as F


def row_normalize(cm: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Row-normalise a confusion matrix so that each row sums to 1.

    Args:
        cm:  Confusion matrix tensor of shape (K, K).
        eps: Small constant for numerical stability.

    Returns:
        Row-normalised confusion matrix of shape (K, K).
    """
    row_sum = cm.sum(dim=1, keepdim=True) + eps
    return cm / row_sum


def update_confusion_matrix(
    cm_counts: torch.Tensor,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    K: int = 8,
) -> torch.Tensor:
    """
    Accumulate batch predictions into a confusion matrix.

    Args:
        cm_counts: Running confusion matrix of shape (K, K).
        y_true:    Ground-truth labels, shape (B,).
        y_pred:    Predicted labels, shape (B,).
        K:         Number of classes.

    Returns:
        Updated confusion matrix.
    """
    for t, p in zip(y_true.view(-1).tolist(), y_pred.view(-1).tolist()):
        if 0 <= t < K and 0 <= p < K:
            cm_counts[t, p] += 1
    return cm_counts


def build_soft_labels(
    cm_row: torch.Tensor,
    y: int,
    delta: float = 0.15,
    temperature: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Construct a confusion-aware soft label distribution for class *y*.

    The true class is boosted by the maximum confusion mass, while classes
    that are frequently confused with *y* (above threshold *delta*) receive
    non-zero weight.  All other classes are zeroed out.

    Formula::

        ỹ[y]   = 1 + max_{j ∈ C_y} CM[y, j]
        ỹ[j]   = CM[y, j]  for j ∈ C_y
        ỹ[j]   = 0          otherwise
        ỹ      = ỹ / Σ_k ỹ_k

    where C_y = { j ≠ y | CM[y, j] > delta }.

    Args:
        cm_row:      Single row of the EMA confusion matrix, shape (K,).
        y:           True class index.
        delta:       Confusion threshold; a class is "confusable" if its
                     confusion rate exceeds this value.
        temperature: Optional temperature for final softmax rescaling.
                     Values < 1 sharpen the distribution; > 1 smooths it.
        eps:         Numerical stability constant.

    Returns:
        Soft label tensor of shape (K,).
    """
    K = cm_row.numel()
    y_tilde = torch.zeros(K, device=cm_row.device, dtype=cm_row.dtype)

    conf_mask = (torch.arange(K, device=cm_row.device) != y) & (cm_row > delta)
    max_conf = cm_row[conf_mask].max() if conf_mask.any() else torch.tensor(0.0, device=cm_row.device)

    y_tilde[y] = 1.0 + float(max_conf)
    y_tilde[conf_mask] = cm_row[conf_mask]
    y_tilde = y_tilde / (y_tilde.sum() + eps)

    if temperature != 1.0:
        y_tilde = F.softmax(torch.log(y_tilde + eps) / temperature, dim=0)

    return y_tilde
