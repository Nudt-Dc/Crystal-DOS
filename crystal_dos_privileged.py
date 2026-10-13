"""Crystal+DOS privileged-information model components without training policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from official_cgcnn_iid import OfficialGraphBatch, load_official_cgcnn_class


class OfficialCgcnnEncoder(nn.Module):
    """Expose the fixed official CGCNN 128-D pre-output representation."""

    def __init__(self, official_model_path: Path | str):
        super().__init__()
        base = load_official_cgcnn_class(official_model_path)(
            92, 41, atom_fea_len=64, n_conv=3, h_fea_len=128,
            n_h=1, classification=False,
        )
        self.embedding = base.embedding
        self.convs = base.convs
        self.pooling = base.pooling
        self.conv_to_fc = base.conv_to_fc
        self.softplus = base.conv_to_fc_softplus

    def forward(self, batch: OfficialGraphBatch) -> torch.Tensor:
        atom = self.embedding(batch.atom_features)
        for convolution in self.convs:
            atom = convolution(atom, batch.neighbor_features, batch.neighbor_indices)
        crystal = self.pooling(atom, batch.crystal_atom_indices)
        return self.softplus(self.conv_to_fc(self.softplus(crystal)))


class DosLatentEncoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4), nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4), nn.SiLU(),
        )
        self.projection = nn.Sequential(nn.Linear(32, latent_dim), nn.SiLU())

    def forward(self, dos: torch.Tensor) -> torch.Tensor:
        if dos.ndim != 3 or dos.shape[1] != 1:
            raise ValueError("DOS must have shape [batch, 1, energy_bins]")
        return self.projection(self.convolutions(dos).mean(dim=-1))


@dataclass(frozen=True)
class PrivilegedOutput:
    prediction: torch.Tensor
    crystal_latent: torch.Tensor
    aligned_crystal_latent: torch.Tensor
    observed_dos_latent: torch.Tensor
    predicted_dos_latent: torch.Tensor
    observed_mask: torch.Tensor


class PrivilegedCrystalDosModel(nn.Module):
    """Use real DOS when present and predicted DOS latent otherwise."""

    def __init__(self, crystal_encoder: nn.Module, crystal_dim: int = 128, dos_dim: int = 64):
        super().__init__()
        self.crystal_encoder = crystal_encoder
        self.dos_encoder = DosLatentEncoder(dos_dim)
        self.dos_predictor = nn.Sequential(
            nn.Linear(crystal_dim, crystal_dim), nn.SiLU(), nn.Linear(crystal_dim, dos_dim)
        )
        self.alignment_projection = nn.Linear(crystal_dim, dos_dim)
        self.property_head = nn.Sequential(
            nn.Linear(crystal_dim + dos_dim + 1, 128), nn.SiLU(), nn.Linear(128, 2)
        )

    def forward(
        self, crystal_batch: OfficialGraphBatch, dos: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> PrivilegedOutput:
        crystal = self.crystal_encoder(crystal_batch)
        observed = self.dos_encoder(dos)
        predicted = self.dos_predictor(crystal)
        aligned_crystal = self.alignment_projection(crystal)
        mask = observed_mask.reshape(-1, 1).to(dtype=crystal.dtype, device=crystal.device)
        if mask.shape[0] != crystal.shape[0] or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("observed_mask must be one binary value per crystal")
        selected = mask * observed + (1 - mask) * predicted
        prediction = self.property_head(torch.cat([crystal, selected, mask], dim=1))
        return PrivilegedOutput(
            prediction, crystal, aligned_crystal, observed, predicted, mask
        )

    def predict_structure_only(self, crystal_batch: OfficialGraphBatch) -> torch.Tensor:
        crystal = self.crystal_encoder(crystal_batch)
        predicted = self.dos_predictor(crystal)
        missing = torch.zeros((crystal.shape[0], 1), dtype=crystal.dtype, device=crystal.device)
        return self.property_head(torch.cat([crystal, predicted, missing], dim=1))


def privileged_losses(
    output: PrivilegedOutput, standardized_targets: torch.Tensor,
    *, property_weight: float = 1.0, alignment_weight: float = 0.1,
    distillation_weight: float = 0.1,
    teacher_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if standardized_targets.shape != output.prediction.shape:
        raise ValueError("targets must match the two-property prediction shape")
    observed = (
        output.observed_mask.reshape(-1).bool() if teacher_mask is None
        else teacher_mask.reshape(-1).to(device=output.prediction.device).bool()
    )
    if observed.shape[0] != output.prediction.shape[0]:
        raise ValueError("teacher_mask must contain one value per crystal")
    property_loss = F.smooth_l1_loss(output.prediction, standardized_targets)
    if observed.any():
        crystal_aligned = F.normalize(output.aligned_crystal_latent[observed], dim=1)
        alignment_loss = 1 - F.cosine_similarity(
            crystal_aligned, F.normalize(output.observed_dos_latent[observed], dim=1), dim=1
        ).mean()
        distillation_loss = F.mse_loss(
            output.predicted_dos_latent[observed],
            output.observed_dos_latent[observed].detach(),
        )
    else:
        alignment_loss = property_loss.new_zeros(())
        distillation_loss = property_loss.new_zeros(())
    total = (
        property_weight * property_loss + alignment_weight * alignment_loss
        + distillation_weight * distillation_loss
    )
    return {"total": total, "property": property_loss,
            "alignment": alignment_loss, "distillation": distillation_loss}
