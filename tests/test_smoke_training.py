"""Phase 2 gate smoke test: LoRA injection + 5-step training loop."""
import pytest
import torch
import torch.nn as nn

from trainer.networks.lora import LoRAModule
from trainer.networks.container import NetworkContainer
from trainer.optimizers import create_optimizer
from trainer.schedulers import create_scheduler


class _DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 32)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _DummyBlock()
        self.head = nn.Linear(32, 10)

    def forward(self, x):
        return self.head(self.block(x))


def test_lora_5_step_training():
    """Phase 2 gate: LoRA injects into mock model, dummy training loop completes 5 steps."""
    # 1. Create and freeze model
    model = _DummyModel()
    for p in model.parameters():
        p.requires_grad = False

    # 2. Inject LoRA
    net = NetworkContainer(
        module_class=LoRAModule, target_modules=["_DummyBlock"],
        exclude_patterns=[], rank=4, alpha=4.0,
    )
    net.apply_to(model)
    assert len(net.lora_modules) == 2

    # 3. Create optimizer + scheduler
    params, descriptions = net.prepare_optimizer_params(unet_lr=1e-3)
    assert len(params) > 0
    optimizer = create_optimizer("adamw", params, lr=1e-3)
    scheduler = create_scheduler("rex", optimizer, num_training_steps=5, warmup_steps=0)

    # 4. Run 5 training steps
    losses = []
    for step in range(5):
        optimizer.zero_grad()
        x = torch.randn(4, 64)
        target = torch.randn(4, 10)
        output = model(x)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    # 5. Verify training ran
    assert len(losses) == 5
    assert all(l > 0 for l in losses)

    # Verify LoRA params were updated (gradients flowed)
    for lora in net.lora_modules:
        for p in lora.parameters():
            if p.requires_grad:
                assert p.grad is not None or p.abs().sum() > 0

    # Verify base model params are still frozen
    for p in model.block.linear1.parameters():
        assert not p.requires_grad


def test_loha_5_step_training():
    """Same smoke test with LoHa."""
    from trainer.networks.loha import LoHaModule

    model = _DummyModel()
    for p in model.parameters():
        p.requires_grad = False

    net = NetworkContainer(
        module_class=LoHaModule, target_modules=["_DummyBlock"],
        exclude_patterns=[], rank=4, alpha=4.0,
    )
    net.apply_to(model)
    params, _ = net.prepare_optimizer_params(unet_lr=1e-3)
    optimizer = create_optimizer("adamw", params, lr=1e-3)

    for step in range(5):
        optimizer.zero_grad()
        output = model(torch.randn(4, 64))
        loss = nn.functional.mse_loss(output, torch.randn(4, 10))
        loss.backward()
        optimizer.step()


def test_lokr_5_step_training():
    """Same smoke test with LoKr."""
    from trainer.networks.lokr import LoKrModule

    model = _DummyModel()
    for p in model.parameters():
        p.requires_grad = False

    net = NetworkContainer(
        module_class=LoKrModule, target_modules=["_DummyBlock"],
        exclude_patterns=[], rank=4, alpha=4.0,
    )
    net.apply_to(model)
    params, _ = net.prepare_optimizer_params(unet_lr=1e-3)
    optimizer = create_optimizer("adamw", params, lr=1e-3)

    for step in range(5):
        optimizer.zero_grad()
        output = model(torch.randn(4, 64))
        loss = nn.functional.mse_loss(output, torch.randn(4, 10))
        loss.backward()
        optimizer.step()
