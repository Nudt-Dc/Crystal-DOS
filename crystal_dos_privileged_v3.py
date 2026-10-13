"""Target-decoupled v3 model and loss for validation-only experiments."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from crystal_dos_privileged import DosLatentEncoder, PrivilegedOutput


class PrivilegedCrystalDosModelV3(nn.Module):
    """Use separate, parameter-matched K and G property heads."""

    def __init__(
        self,
        crystal_encoder: nn.Module,
        crystal_dim: int = 128,
        dos_dim: int = 64,
        head_hidden_dim: int = 64,
    ):
        super().__init__()
        self.crystal_encoder = crystal_encoder
        self.dos_encoder = DosLatentEncoder(dos_dim)
        self.dos_predictor = nn.Sequential(
            nn.Linear(crystal_dim, crystal_dim),
            nn.SiLU(),
            nn.Linear(crystal_dim, dos_dim),
        )
        self.alignment_projection = nn.Linear(crystal_dim, dos_dim)
        head_input_dim = crystal_dim + dos_dim + 1
        self.property_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(head_input_dim, head_hidden_dim),
                nn.SiLU(),
                nn.Linear(head_hidden_dim, 1),
            )
            for _ in range(2)
        )

    def _predict(
        self, crystal: torch.Tensor, selected: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat([crystal, selected, mask], dim=1)
        return torch.cat([head(features) for head in self.property_heads], dim=1)

    def forward(
        self,
        crystal_batch,
        dos: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> PrivilegedOutput:
        crystal = self.crystal_encoder(crystal_batch)
        observed = self.dos_encoder(dos)
        predicted = self.dos_predictor(crystal)
        aligned_crystal = self.alignment_projection(crystal)
        mask = observed_mask.reshape(-1, 1).to(
            dtype=crystal.dtype, device=crystal.device
        )
        if mask.shape[0] != crystal.shape[0] or not torch.all(
            (mask == 0) | (mask == 1)
        ):
            raise ValueError("observed_mask must be one binary value per crystal")
        selected = mask * observed + (1 - mask) * predicted
        return PrivilegedOutput(
            self._predict(crystal, selected, mask),
            crystal,
            aligned_crystal,
            observed,
            predicted,
            mask,
        )

    def predict_structure_only(self, crystal_batch) -> torch.Tensor:
        crystal = self.crystal_encoder(crystal_batch)
        predicted = self.dos_predictor(crystal)
        missing = torch.zeros(
            (crystal.shape[0], 1), dtype=crystal.dtype, device=crystal.device
        )
        return self._predict(crystal, predicted, missing)


def privileged_losses_v3(
    output: PrivilegedOutput,
    standardized_targets: torch.Tensor,
    *,
    target_weights: Sequence[float] = (1.0, 1.0),
    alignment_weight: float = 0.02,
    distillation_weight: float = 0.1,
    teacher_mask: torch.Tensor | None = None,
    temperature: float = 0.1,
) -> dict[str, torch.Tensor]:
    if standardized_targets.shape != output.prediction.shape:
        raise ValueError("targets must match the two-property prediction shape")
    weights = torch.as_tensor(
        target_weights, dtype=output.prediction.dtype, device=output.prediction.device
    )
    if weights.shape != (2,) or not torch.isfinite(weights).all() or not torch.all(
        weights > 0
    ):
        raise ValueError("target_weights must contain two finite positive values")
    if not 0 < float(temperature):
        raise ValueError("temperature must be positive")
    weights = weights / weights.mean()
    property_by_target = F.smooth_l1_loss(
        output.prediction, standardized_targets, reduction="none"
    ).mean(dim=0)
    property_loss = (property_by_target * weights).mean()
    observed = (
        output.observed_mask.reshape(-1).bool()
        if teacher_mask is None
        else teacher_mask.reshape(-1).to(device=output.prediction.device).bool()
    )
    if observed.shape[0] != output.prediction.shape[0]:
        raise ValueError("teacher_mask must contain one value per crystal")
    if int(observed.sum()) >= 2:
        crystal = F.normalize(output.aligned_crystal_latent[observed], dim=1)
        dos = F.normalize(output.observed_dos_latent[observed], dim=1)
        logits = crystal @ dos.T / float(temperature)
        labels = torch.arange(len(logits), device=logits.device)
        alignment_loss = 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )
    else:
        alignment_loss = property_loss.new_zeros(())
    if observed.any():
        distillation_loss = F.mse_loss(
            output.predicted_dos_latent[observed],
            output.observed_dos_latent[observed].detach(),
        )
    else:
        distillation_loss = property_loss.new_zeros(())
    total = (
        property_loss
        + alignment_weight * alignment_loss
        + distillation_weight * distillation_loss
    )
    return {
        "total": total,
        "property": property_loss,
        "property_k": property_by_target[0],
        "property_g": property_by_target[1],
        "alignment": alignment_loss,
        "distillation": distillation_loss,
    }
