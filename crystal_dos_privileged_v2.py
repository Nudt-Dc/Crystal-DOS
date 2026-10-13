"""Corrective v2 contrastive objective; v1 model code remains immutable."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from crystal_dos_privileged import PrivilegedCrystalDosModel, PrivilegedOutput


def privileged_losses_v2(
    output: PrivilegedOutput, standardized_targets: torch.Tensor, *,
    property_weight: float = 1.0, alignment_weight: float = 0.02,
    distillation_weight: float = 0.1, teacher_mask: torch.Tensor | None = None,
    temperature: float = 0.1,
) -> dict[str, torch.Tensor]:
    if standardized_targets.shape != output.prediction.shape:
        raise ValueError("targets must match the two-property prediction shape")
    if not 0 < float(temperature):
        raise ValueError("temperature must be positive")
    observed = (
        output.observed_mask.reshape(-1).bool() if teacher_mask is None
        else teacher_mask.reshape(-1).to(device=output.prediction.device).bool()
    )
    if observed.shape[0] != output.prediction.shape[0]:
        raise ValueError("teacher_mask must contain one value per crystal")
    property_loss = F.smooth_l1_loss(output.prediction, standardized_targets)
    if int(observed.sum()) >= 2:
        crystal = F.normalize(output.aligned_crystal_latent[observed], dim=1)
        dos = F.normalize(output.observed_dos_latent[observed], dim=1)
        logits = crystal @ dos.T / float(temperature)
        labels = torch.arange(len(logits), device=logits.device)
        alignment_loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    else:
        alignment_loss = property_loss.new_zeros(())
    if observed.any():
        distillation_loss = F.mse_loss(
            output.predicted_dos_latent[observed], output.observed_dos_latent[observed].detach()
        )
    else:
        distillation_loss = property_loss.new_zeros(())
    total = property_weight * property_loss + alignment_weight * alignment_loss + distillation_weight * distillation_loss
    return {"total": total, "property": property_loss, "alignment": alignment_loss, "distillation": distillation_loss}
