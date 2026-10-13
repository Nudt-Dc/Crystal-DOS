import pytest
import torch
from torch import nn

from crystal_dos_privileged import PrivilegedCrystalDosModel, PrivilegedOutput
from crystal_dos_privileged_v3 import (
    PrivilegedCrystalDosModelV3,
    privileged_losses_v3,
)


def _output(prediction: torch.Tensor) -> PrivilegedOutput:
    rows = prediction.shape[0]
    latent = torch.eye(rows)
    return PrivilegedOutput(
        prediction=prediction,
        crystal_latent=torch.zeros((rows, 128)),
        aligned_crystal_latent=latent,
        observed_dos_latent=latent,
        predicted_dos_latent=latent.clone(),
        observed_mask=torch.ones((rows, 1)),
    )


def _head_parameters(model: nn.Module) -> int:
    modules = (
        [model.property_head]
        if hasattr(model, "property_head")
        else list(model.property_heads)
    )
    return sum(parameter.numel() for module in modules for parameter in module.parameters())


def test_decoupled_heads_are_parameter_matched_to_v2_shared_head():
    shared = PrivilegedCrystalDosModel(nn.Identity())
    decoupled = PrivilegedCrystalDosModelV3(nn.Identity())
    relative_difference = abs(_head_parameters(shared) - _head_parameters(decoupled)) / _head_parameters(shared)
    assert relative_difference < 0.02


def test_equal_target_weights_match_unweighted_smooth_l1():
    prediction = torch.tensor([[0.0, 0.5], [1.0, -0.5]])
    targets = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    result = privileged_losses_v3(_output(prediction), targets)
    expected = torch.nn.functional.smooth_l1_loss(prediction, targets)
    assert torch.allclose(result["property"], expected)


def test_k125_weight_increases_k_dominated_property_loss_without_rescaling_sum():
    prediction = torch.zeros((2, 2))
    targets = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    equal = privileged_losses_v3(_output(prediction), targets)
    weighted = privileged_losses_v3(
        _output(prediction), targets, target_weights=(1.25, 0.75)
    )
    assert weighted["property"] > equal["property"]
    assert weighted["property_k"] == equal["property_k"]
    assert weighted["property_g"] == equal["property_g"]


@pytest.mark.parametrize("weights", [(1.0,), (1.0, 0.0), (1.0, float("nan"))])
def test_target_weights_fail_closed(weights):
    with pytest.raises(ValueError, match="two finite positive"):
        privileged_losses_v3(_output(torch.zeros((2, 2))), torch.zeros((2, 2)), target_weights=weights)
