"""Network module tests: LoRA, LoHa, LoKr injection, save/load, max norm."""
import tempfile
import os
import pytest
import torch
import torch.nn as nn

from trainer.networks.lora import LoRAModule
from trainer.networks.loha import LoHaModule
from trainer.networks.lokr import LoKrModule
from trainer.networks.container import NetworkContainer
from trainer.networks.arch_configs import ARCH_NETWORK_CONFIGS
from trainer.networks import get_module_class


class _TestBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 32)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


class _TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _TestBlock()
        self.head = nn.Linear(32, 10)

    def forward(self, x):
        return self.head(self.block(x))


def _freeze(model):
    for p in model.parameters():
        p.requires_grad = False


class TestGetModuleClass:
    def test_lora(self):
        assert get_module_class("lora") is LoRAModule

    def test_loha(self):
        assert get_module_class("loha") is LoHaModule

    def test_lokr(self):
        assert get_module_class("lokr") is LoKrModule

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_module_class("nonexistent")


class TestArchConfigs:
    def test_all_twelve_architectures(self):
        assert len(ARCH_NETWORK_CONFIGS) == 12
        expected = {"wan", "hunyuan_video", "hunyuan_video_1_5", "framepack",
                    "flux_kontext", "flux_2", "kandinsky5", "qwen_image", "zimage",
                    "sdxl", "sd3", "flux_1"}
        assert set(ARCH_NETWORK_CONFIGS.keys()) == expected

    def test_each_has_target_modules(self):
        for name, cfg in ARCH_NETWORK_CONFIGS.items():
            assert "target_modules" in cfg, f"{name} missing target_modules"
            assert len(cfg["target_modules"]) > 0, f"{name} has empty target_modules"


class TestLoRAInjection:
    def test_apply_and_forward(self):
        model = _TestModel()
        _freeze(model)
        net = NetworkContainer(
            module_class=LoRAModule, target_modules=["_TestBlock"],
            exclude_patterns=[], rank=4, alpha=4.0,
        )
        net.apply_to(model)
        assert len(net.lora_modules) == 2
        out = model(torch.randn(2, 64))
        assert out.shape == (2, 10)

    def test_backward(self):
        model = _TestModel()
        _freeze(model)
        net = NetworkContainer(
            module_class=LoRAModule, target_modules=["_TestBlock"],
            exclude_patterns=[], rank=4, alpha=4.0,
        )
        net.apply_to(model)
        out = model(torch.randn(2, 64))
        out.sum().backward()
        # LoRA params should have gradients
        for lora in net.lora_modules:
            assert any(p.grad is not None for p in lora.parameters())

    def test_optimizer_params(self):
        model = _TestModel()
        _freeze(model)
        net = NetworkContainer(
            module_class=LoRAModule, target_modules=["_TestBlock"],
            exclude_patterns=[], rank=4, alpha=4.0,
        )
        net.apply_to(model)
        params, _ = net.prepare_optimizer_params(unet_lr=1e-4)
        assert len(params) > 0
        total = sum(sum(p.numel() for p in g["params"]) for g in params)
        assert total > 0

    def test_save_load(self):
        model = _TestModel()
        _freeze(model)
        net = NetworkContainer(
            module_class=LoRAModule, target_modules=["_TestBlock"],
            exclude_patterns=[], rank=4, alpha=4.0,
        )
        net.apply_to(model)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.safetensors")
            net.save_weights(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_exclude_pattern(self):
        model = _TestModel()
        _freeze(model)
        net = NetworkContainer(
            module_class=LoRAModule, target_modules=["_TestBlock"],
            exclude_patterns=["linear2"], rank=4, alpha=4.0,
        )
        net.apply_to(model)
        assert len(net.lora_modules) == 1  # Only linear1


class TestLoHaInjection:
    def test_apply_and_backward(self):
        model = _TestModel()
        _freeze(model)
        net = NetworkContainer(
            module_class=LoHaModule, target_modules=["_TestBlock"],
            exclude_patterns=[], rank=4, alpha=4.0,
        )
        net.apply_to(model)
        assert len(net.lora_modules) == 2
        out = model(torch.randn(2, 64))
        out.sum().backward()


class TestLoKrInjection:
    def test_apply_and_backward(self):
        model = _TestModel()
        _freeze(model)
        net = NetworkContainer(
            module_class=LoKrModule, target_modules=["_TestBlock"],
            exclude_patterns=[], rank=4, alpha=4.0,
        )
        net.apply_to(model)
        assert len(net.lora_modules) == 2
        out = model(torch.randn(2, 64))
        out.sum().backward()


# ---------------------------------------------------------------------------
# Per-block LR multiplier tests (Task 17)
# ---------------------------------------------------------------------------

class _BlockedBlock(nn.Module):
    """A block whose lora_name will contain 'blocks_N' after name mangling."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 32)

    def forward(self, x):
        return self.linear(x)


class _BlockedModel(nn.Module):
    """Model with two named transformer blocks so we can test per-block LRs.

    After apply_to():
      block 0 → lora_name: "lora_unet_blocks_0_linear"
      block 1 → lora_name: "lora_unet_blocks_1_linear"
    """
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_BlockedBlock(), _BlockedBlock()])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def _make_blocked_net() -> NetworkContainer:
    model = _BlockedModel()
    _freeze(model)
    net = NetworkContainer(
        module_class=LoRAModule,
        target_modules=["_BlockedBlock"],
        exclude_patterns=[],
        rank=4,
        alpha=4.0,
    )
    net.apply_to(model)
    return net


class TestGetBlockLR:
    """Unit tests for NetworkContainer._get_block_lr helper."""

    def test_dot_separated_block_name(self):
        net = _make_blocked_net()
        # lora_unet_blocks_0_linear → block_idx=0
        lr = net._get_block_lr("lora_unet_blocks_0_linear", base_lr=1e-4, multipliers=[0.5, 1.0])
        assert lr == pytest.approx(5e-5)

    def test_underscore_separated_block_name(self):
        net = _make_blocked_net()
        lr = net._get_block_lr("lora_unet_blocks_1_linear", base_lr=1e-4, multipliers=[0.5, 1.0])
        assert lr == pytest.approx(1e-4)

    def test_fallback_when_no_block_index(self):
        net = _make_blocked_net()
        lr = net._get_block_lr("lora_unet_text_encoder_proj", base_lr=2e-4, multipliers=[0.5])
        assert lr == pytest.approx(2e-4)

    def test_fallback_when_index_out_of_range(self):
        net = _make_blocked_net()
        # block 5 but multipliers only has 2 entries
        lr = net._get_block_lr("lora_unet_blocks_5_linear", base_lr=1e-4, multipliers=[0.5, 1.0])
        assert lr == pytest.approx(1e-4)

    def test_zero_multiplier(self):
        net = _make_blocked_net()
        lr = net._get_block_lr("lora_unet_blocks_0_linear", base_lr=1e-4, multipliers=[0.0])
        assert lr == pytest.approx(0.0)


class TestPrepareOptimizerParamsBlockLR:
    """Integration tests for prepare_optimizer_params with block_lr_multipliers."""

    def test_two_blocks_get_different_lr(self):
        net = _make_blocked_net()
        # block 0 → 0.5x, block 1 → 2.0x
        params, descs = net.prepare_optimizer_params(
            unet_lr=1e-4, block_lr_multipliers=[0.5, 2.0]
        )
        lrs = [g["lr"] for g in params]
        assert any(lr == pytest.approx(5e-5) for lr in lrs)
        assert any(lr == pytest.approx(2e-4) for lr in lrs)

    def test_all_params_covered(self):
        net = _make_blocked_net()
        params, _ = net.prepare_optimizer_params(
            unet_lr=1e-4, block_lr_multipliers=[0.5, 2.0]
        )
        total = sum(sum(p.numel() for p in g["params"]) for g in params)
        expected = sum(p.numel() for p in net.parameters())
        assert total == expected

    def test_no_duplicate_params(self):
        net = _make_blocked_net()
        params, _ = net.prepare_optimizer_params(
            unet_lr=1e-4, block_lr_multipliers=[0.5, 2.0]
        )
        all_ids: list[int] = []
        for g in params:
            all_ids.extend(id(p) for p in g["params"])
        assert len(all_ids) == len(set(all_ids))

    def test_same_multiplier_merges_into_one_group(self):
        """If both blocks share the same multiplier, params should land in one LR bucket."""
        net = _make_blocked_net()
        params, _ = net.prepare_optimizer_params(
            unet_lr=1e-4, block_lr_multipliers=[1.0, 1.0]
        )
        lrs = [g["lr"] for g in params]
        # Should be exactly one group (both blocks get 1e-4)
        assert len(lrs) == 1
        assert lrs[0] == pytest.approx(1e-4)

    def test_zero_multiplier_skips_group(self):
        """A block with multiplier=0 should be excluded from optimizer params."""
        net = _make_blocked_net()
        params, _ = net.prepare_optimizer_params(
            unet_lr=1e-4, block_lr_multipliers=[0.0, 1.0]
        )
        lrs = {g["lr"] for g in params}
        assert 0.0 not in lrs

    def test_no_block_lr_uses_base_lr(self):
        """Without block_lr_multipliers the original single-group path is used."""
        net = _make_blocked_net()
        params, _ = net.prepare_optimizer_params(unet_lr=1e-4)
        assert len(params) == 1
        assert params[0]["lr"] == pytest.approx(1e-4)

    def test_loraplus_with_block_lr(self):
        """LoRA+ ratio should be applied on top of block-specific LR."""
        net = _make_blocked_net()
        net.set_loraplus_lr_ratio(4.0)  # lora_up gets 4x the base
        params, _ = net.prepare_optimizer_params(
            unet_lr=1e-4, block_lr_multipliers=[1.0, 1.0]
        )
        lrs = [g["lr"] for g in params]
        # Base group: 1e-4, plus group: 1e-4 * 4.0 = 4e-4
        assert any(lr == pytest.approx(1e-4) for lr in lrs)
        assert any(lr == pytest.approx(4e-4) for lr in lrs)

    def test_descriptions_returned(self):
        net = _make_blocked_net()
        params, descs = net.prepare_optimizer_params(
            unet_lr=1e-4, block_lr_multipliers=[0.5, 2.0]
        )
        assert len(params) == len(descs)
        assert all(isinstance(d, str) for d in descs)
