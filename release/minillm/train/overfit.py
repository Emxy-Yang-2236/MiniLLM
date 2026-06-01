from __future__ import annotations

import torch

from minillm.train.optim import AdamW, clip_grad_norm_


def tiny_overfit_batch(
    model,
    batch: dict[str, torch.Tensor],
    steps: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    max_grad_norm: float | None = 1.0,
    device: str | torch.device = "cpu",
) -> dict[str, list[float] | float]:
    """Provided diagnostic helper used by public tests and debugging.

    Students implement the model, optimizer, clipping, and SFT data path. This
    helper only wires those pieces together on one tiny batch.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    dev = torch.device(device)
    model.to(dev)
    model.train()
    input_ids = batch["input_ids"].to(dev)
    labels = batch["labels"].to(dev)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses: list[float] = []
    grad_norms: list[float] = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = model(input_ids, labels)["loss"]
        loss.backward()
        if max_grad_norm is None:
            total = torch.zeros((), device=dev)
            for param in model.parameters():
                if param.grad is not None:
                    total += torch.sum(param.grad.detach() ** 2)
            grad_norm = torch.sqrt(total)
        else:
            grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        grad_norms.append(float(grad_norm.detach().cpu()))
    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "best_loss": min(losses),
        "losses": losses,
        "grad_norms": grad_norms,
    }
