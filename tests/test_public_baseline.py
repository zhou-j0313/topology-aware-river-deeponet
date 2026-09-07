import importlib.util
import sys
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "src" / "river_deeponet.py"
SPEC = importlib.util.spec_from_file_location("river_deeponet", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_masked_mse_ignores_unobserved_values():
    prediction = torch.tensor([[1.0, 100.0]])
    target = torch.tensor([[0.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0]])
    loss = MODULE.RiverOperatorSurrogate._masked_mse(
        prediction, target, mask)
    assert torch.isclose(loss, torch.tensor(1.0))


def test_public_consistency_loss_is_differentiable():
    surrogate = MODULE.RiverOperatorSurrogate.__new__(
        MODULE.RiverOperatorSurrogate)
    torch.nn.Module.__init__(surrogate)
    surrogate.CHANNELS = {"time_mask": 0, "tau": 1, "x_local_norm": 2}
    surrogate.reach_names = ["reach_1"]
    surrogate.reach_row_indices = {"reach_1": [0, 1, 2]}

    prediction = torch.randn(2, 2, 3, 5, requires_grad=True)
    inputs = torch.zeros(2, 3, 3, 5)
    inputs[:, 0] = 1.0
    inputs[:, 1] = torch.linspace(0.0, 1.0, 5)
    inputs[:, 2, :, 0] = torch.tensor([0.0, 0.5, 1.0])

    continuity, momentum = surrogate._public_consistency_loss(
        prediction, inputs)
    (continuity + momentum).backward()

    assert torch.isfinite(continuity)
    assert torch.isfinite(momentum)
    assert torch.isfinite(prediction.grad).all()
