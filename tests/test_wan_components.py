"""Milestone 1 tests: Wan component imports and basic validation."""
import torch
import pytest


class TestWanConfigs:
    def test_all_task_keys(self):
        from trainer.arch.wan.components.configs import WAN_CONFIGS
        expected = {
            "t2v-14B", "t2v-1.3B", "i2v-14B", "t2i-14B", "flf2v-14B",
            "t2v-1.3B-FC", "t2v-14B-FC", "i2v-14B-FC",
            "i2v-A14B", "t2v-A14B",
        }
        assert set(WAN_CONFIGS.keys()) == expected

    def test_t2v_14B_fields(self):
        from trainer.arch.wan.components.configs import WAN_CONFIGS
        cfg = WAN_CONFIGS["t2v-14B"]
        assert cfg.dim == 5120
        assert cfg.num_layers == 40
        assert cfg.num_heads == 40
        assert cfg.in_dim == 16
        assert cfg.patch_size == (1, 2, 2)
        assert cfg.vae_stride == (4, 8, 8)
        assert cfg.i2v is False
        assert cfg.v2_2 is False

    def test_i2v_14B_has_clip_fields(self):
        from trainer.arch.wan.components.configs import WAN_CONFIGS
        cfg = WAN_CONFIGS["i2v-14B"]
        assert cfg.i2v is True
        assert cfg.in_dim == 36
        assert hasattr(cfg, "clip_model")
        assert hasattr(cfg, "clip_dtype")

    def test_fun_control_in_dim(self):
        from trainer.arch.wan.components.configs import WAN_CONFIGS
        assert WAN_CONFIGS["t2v-14B-FC"].in_dim == 48
        assert WAN_CONFIGS["t2v-14B-FC"].is_fun_control is True

    def test_wan_2_2_has_boundary(self):
        from trainer.arch.wan.components.configs import WAN_CONFIGS
        assert WAN_CONFIGS["i2v-A14B"].v2_2 is True
        assert WAN_CONFIGS["i2v-A14B"].boundary is not None
        assert WAN_CONFIGS["t2v-A14B"].v2_2 is True


class TestAttention:
    def test_sdpa_forward(self):
        """flash_attention with torch (SDPA) mode works on small CPU tensors."""
        from trainer.arch.wan.components.attention import flash_attention
        # flash_attention expects [B, L, N, C] format (not [B, N, L, C])
        B, L, N, D = 1, 8, 4, 16
        q = torch.randn(B, L, N, D)
        k = torch.randn(B, L, N, D)
        v = torch.randn(B, L, N, D)
        out = flash_attention([q, k, v], attn_mode="torch")
        assert out.shape == (B, L, N, D)


class TestRegistry:
    def test_registry_discovers_wan(self):
        from trainer.registry import list_models
        assert "wan" in list_models()

    def test_registry_resolves_wan(self):
        from trainer.registry import get_model_strategy
        cls = get_model_strategy("wan")
        assert cls.__name__ == "WanStrategy"

    def test_wan_strategy_properties(self):
        from trainer.registry import get_model_strategy
        from trainer.config.schema import TrainConfig, ModelConfig
        cls = get_model_strategy("wan")
        config = TrainConfig(
            model=ModelConfig(architecture="wan", base_model_path="/fake"),
            training={"method": "full_finetune"},
            data={"dataset_config_path": "/fake.toml"},
        )
        strategy = cls(config)
        assert strategy.architecture == "wan"
        assert strategy.supports_video is True
