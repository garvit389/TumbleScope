import pytest
import torch
from torch import nn

from core.config import load_config
from ml.model import PeriodCNN


@pytest.fixture
def model():
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            yield PeriodCNN(load_config().cnn)
    finally:
        torch.set_num_threads(threads)


def batch():
    windows = torch.zeros(4, 3, 1024)
    t = torch.arange(1024)
    for row, cycle in enumerate((8, 24, 96)):
        windows[row, 0] = torch.cos(2 * torch.pi * t / cycle)
    windows[:, 1] = 1
    windows[:, 2, 1:] = .5
    target = torch.zeros(4, 129)
    for row, index in enumerate((5, 38, 77)):
        target[row, index:index + 2] = torch.tensor([.8, .2])
    target[3, 128] = 1
    return windows, target


def test_structure_and_gradients(model):
    expected_widths = (32, 32, 64, 64, 128, 128)
    widths = [block.conv.out_channels for block in model.blocks]
    assert tuple(widths) == expected_widths
    rf = 1
    for i, block in enumerate(model.blocks):
        conv = block.conv
        assert conv.kernel_size == (7,) and conv.dilation == (2**i,) and conv.stride == (1,)
        assert conv.padding == (3 * 2**i,) and conv.bias is None
        assert isinstance(block.norm, nn.BatchNorm1d)
        projections = int(conv.in_channels != conv.out_channels)
        assert sum(isinstance(m, nn.Conv1d) for m in block.modules()) == 1 + projections
        if projections:
            assert block.skip.kernel_size == (1,) and block.skip.bias is None
        rf += (conv.kernel_size[0] - 1) * conv.dilation[0]
    assert rf == 379
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 250753
    x, target = batch()
    logits = model(x)
    assert logits.shape == (4, 129) and torch.isfinite(logits).all()
    nn.functional.cross_entropy(logits, target).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.blocks[0].conv.weight.grad.abs().sum() > 0
    assert model.head.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="1024"):
        model(torch.zeros(2, 3, 64))


def test_overfit_one_batch(model):
    x, target = batch()
    floor = -(target * target.clamp_min(1e-10).log()).sum(1).mean().item()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)

    @torch.no_grad()
    def evaluate():
        model.eval()
        logits = model(x)
        return (nn.functional.cross_entropy(logits, target).item(),
                torch.equal(logits.argmax(1), target.argmax(1)))

    initial, _ = evaluate()
    for step in range(1, 301):
        model.train()
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x), target)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        final, memorized = evaluate()
        if memorized and final - floor <= .5 * (initial - floor):
            break
    print(f"seed=17 steps={step} initial={initial:.6f} final={final:.6f} entropy={floor:.6f}")
    assert memorized and final - floor <= .5 * (initial - floor)
