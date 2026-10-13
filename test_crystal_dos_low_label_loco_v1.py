from types import SimpleNamespace

import numpy as np
import torch

from crystal_dos_low_label_loco_v1 import (
    compute_low_label_normalizers,
    low_label_losses,
)


def test_target_normalizer_uses_only_labelled_training_rows():
    data = SimpleNamespace(
        partitions={"train": np.array([0, 1, 2, 3])},
        paired_partitions={"train": np.array([0, 1, 2, 3])},
        dos_features=np.arange(16, dtype=np.float32).reshape(4, 1, 4),
        dos_rows=np.arange(4),
        targets=np.array([[1.0, 10.0], [3.0, 30.0], [1000.0, 2000.0], [2000.0, 4000.0]]),
    )
    membership = np.array([1, 1, 0, 0], dtype=np.uint8)
    normalizers = compute_low_label_normalizers(data, membership)
    np.testing.assert_allclose(normalizers.target_mean, [2.0, 20.0])
    np.testing.assert_allclose(normalizers.target_scale, np.std(data.targets[:2], axis=0, ddof=1))


def _output(prediction):
    return SimpleNamespace(
        prediction=prediction,
        aligned_crystal_latent=torch.eye(3),
        observed_dos_latent=torch.eye(3),
        predicted_dos_latent=torch.eye(3),
    )


def test_unlabelled_targets_do_not_change_property_or_total_loss():
    prediction = torch.tensor([[0.0, 0.0], [2.0, 2.0], [1.0, 1.0]], requires_grad=True)
    targets_a = torch.tensor([[1.0, 1.0], [10.0, 20.0], [0.0, 0.0]])
    targets_b = torch.tensor([[1.0, 1.0], [-999.0, 999.0], [0.0, 0.0]])
    label_mask = torch.tensor([1, 0, 1])
    teacher_mask = torch.tensor([1, 1, 1])
    kwargs = dict(
        alignment_weight=0.02,
        distillation_weight=0.1,
        teacher_mask=teacher_mask,
    )
    first = low_label_losses(_output(prediction), targets_a, label_mask, **kwargs)
    second = low_label_losses(_output(prediction), targets_b, label_mask, **kwargs)
    assert torch.equal(first["property"], second["property"])
    assert torch.equal(first["total"], second["total"])


def test_zero_label_batch_keeps_differentiable_zero_property_loss():
    prediction = torch.ones((3, 2), requires_grad=True)
    losses = low_label_losses(
        _output(prediction),
        torch.zeros((3, 2)),
        torch.zeros(3),
        alignment_weight=0.0,
        distillation_weight=0.0,
        teacher_mask=torch.zeros(3),
    )
    losses["total"].backward()
    assert losses["property"].item() == 0.0
    assert prediction.grad is not None

