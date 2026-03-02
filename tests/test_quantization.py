"""Tests for trainer.quantization package."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Module-level helper — must be defined BEFORE any test class that uses it
# ---------------------------------------------------------------------------

def _bnb_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# TestQuantizationRegistry
# ---------------------------------------------------------------------------

class TestQuantizationRegistry:
    """Tests for get_quantizer() registration and lookup."""

    def test_fp8_returns_callable(self):
        from trainer.quantization import get_quantizer
        factory = get_quantizer("fp8")
        assert callable(factory)

    def test_fp8_scaled_returns_callable(self):
        from trainer.quantization import get_quantizer
        factory = get_quantizer("fp8_scaled")
        assert callable(factory)

    def test_nf4_returns_callable(self):
        from trainer.quantization import get_quantizer
        factory = get_quantizer("nf4")
        assert callable(factory)

    def test_int8_returns_callable(self):
        from trainer.quantization import get_quantizer
        factory = get_quantizer("int8")
        assert callable(factory)

    def test_none_returns_none(self):
        from trainer.quantization import get_quantizer
        result = get_quantizer(None)
        assert result is None

    def test_invalid_raises_value_error(self):
        from trainer.quantization import get_quantizer
        with pytest.raises(ValueError, match="Unknown quantization type"):
            get_quantizer("q4_0")

    def test_invalid_error_lists_supported(self):
        from trainer.quantization import get_quantizer
        with pytest.raises(ValueError, match="fp8"):
            get_quantizer("unknown_type")


# ---------------------------------------------------------------------------
# TestQuantizationConfig
# ---------------------------------------------------------------------------

class TestQuantizationConfig:
    """Tests that TrainConfig accepts all quantization types on ModelConfig."""

    _BASE_KWARGS = {
        "model": {
            "architecture": "wan",
            "base_model_path": "/fake/model.safetensors",
        },
        "training": {"method": "full_finetune"},
        "data": {"dataset_config_path": "/fake.toml"},
    }

    def _make_config(self, quantization: str | None):
        from trainer.config.schema import TrainConfig
        kwargs = {
            **self._BASE_KWARGS,
            "model": {
                **self._BASE_KWARGS["model"],
                "quantization": quantization,
            },
        }
        return TrainConfig(**kwargs)

    def test_none_quantization(self):
        cfg = self._make_config(None)
        assert cfg.model.quantization is None

    def test_nf4_quantization(self):
        cfg = self._make_config("nf4")
        assert cfg.model.quantization == "nf4"

    def test_int8_quantization(self):
        cfg = self._make_config("int8")
        assert cfg.model.quantization == "int8"

    def test_fp8_quantization(self):
        cfg = self._make_config("fp8")
        assert cfg.model.quantization == "fp8"

    def test_fp8_scaled_quantization(self):
        cfg = self._make_config("fp8_scaled")
        assert cfg.model.quantization == "fp8_scaled"


# ---------------------------------------------------------------------------
# TestFP8Quantization
# ---------------------------------------------------------------------------

class TestFP8Quantization:
    """Tests for LinearFp8 and quantize_linear_fp8."""

    def _make_linear(self, in_f=32, out_f=16, bias=True) -> nn.Linear:
        lin = nn.Linear(in_f, out_f, bias=bias)
        return lin

    def test_weight_dtype_is_fp8(self):
        from trainer.quantization.fp8 import quantize_linear_fp8
        lin = self._make_linear()
        q = quantize_linear_fp8(lin)
        assert q.weight.dtype == torch.float8_e4m3fn

    def test_preserves_output_shape(self):
        from trainer.quantization.fp8 import quantize_linear_fp8
        lin = self._make_linear(in_f=32, out_f=16)
        q = quantize_linear_fp8(lin)
        x = torch.randn(4, 32)
        out = q(x)
        assert out.shape == (4, 16)

    def test_weight_shape_preserved(self):
        from trainer.quantization.fp8 import quantize_linear_fp8
        lin = self._make_linear(in_f=32, out_f=16)
        q = quantize_linear_fp8(lin)
        assert q.weight.shape == (16, 32)

    def test_scaled_stores_weight_scale(self):
        from trainer.quantization.fp8 import quantize_linear_fp8
        lin = self._make_linear()
        q = quantize_linear_fp8(lin, scaled=True)
        assert q._weight_scale is not None
        assert q._weight_scale.numel() == 1

    def test_unscaled_no_weight_scale(self):
        from trainer.quantization.fp8 import quantize_linear_fp8
        lin = self._make_linear()
        q = quantize_linear_fp8(lin, scaled=False)
        assert q._weight_scale is None

    def test_weights_frozen_after_quantize(self):
        from trainer.quantization.fp8 import quantize_linear_fp8
        lin = self._make_linear()
        q = quantize_linear_fp8(lin)
        assert not q.weight.requires_grad

    def test_bias_frozen_after_quantize(self):
        from trainer.quantization.fp8 import quantize_linear_fp8
        lin = self._make_linear(bias=True)
        q = quantize_linear_fp8(lin)
        assert not q.bias.requires_grad

    def test_no_bias_when_linear_has_no_bias(self):
        from trainer.quantization.fp8 import quantize_linear_fp8
        lin = self._make_linear(bias=False)
        q = quantize_linear_fp8(lin)
        assert q.bias is None

    def test_forward_compute_dtype(self):
        from trainer.quantization.fp8 import quantize_linear_fp8
        lin = self._make_linear(in_f=32, out_f=16)
        q = quantize_linear_fp8(lin, compute_dtype=torch.bfloat16)
        x = torch.randn(2, 32)
        out = q(x)
        # Output should be cast to compute_dtype
        assert out.dtype == torch.bfloat16

    def test_dequantize_weight_returns_float(self):
        from trainer.quantization.fp8 import quantize_linear_fp8
        lin = self._make_linear()
        q = quantize_linear_fp8(lin, scaled=True)
        dq = q.dequantize_weight()
        assert dq.dtype == torch.float32

    def test_dequantize_weight_shape(self):
        from trainer.quantization.fp8 import quantize_linear_fp8
        lin = self._make_linear(in_f=32, out_f=16)
        q = quantize_linear_fp8(lin)
        dq = q.dequantize_weight()
        assert dq.shape == (16, 32)

    def test_from_linear_class_method(self):
        from trainer.quantization.fp8 import LinearFp8
        lin = self._make_linear()
        q = LinearFp8.from_linear(lin)
        assert isinstance(q, LinearFp8)


# ---------------------------------------------------------------------------
# TestQuantizeModel
# ---------------------------------------------------------------------------

class TestQuantizeModel:
    """Tests for quantize_model() — graph walking and stats."""

    def _make_model_with_norms(self):
        """Model with linears, a LayerNorm, and an Embedding."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(10, 16)
                self.norm = nn.LayerNorm(16)
                self.fc1 = nn.Linear(16, 32)
                self.fc2 = nn.Linear(32, 16)

            def forward(self, x):
                return self.fc2(self.fc1(x))

        return TestModel()

    def _make_nested_model(self):
        """Nested model to test recursive walking."""
        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 8)

        class Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Inner()
                self.top_linear = nn.Linear(8, 4)

        return Outer()

    def test_skips_layernorm(self):
        from trainer.quantization import quantize_model
        model = self._make_model_with_norms()
        stats = quantize_model(model, "fp8")
        assert stats["skipped"] >= 1
        assert isinstance(model.norm, nn.LayerNorm)

    def test_skips_embedding(self):
        from trainer.quantization import quantize_model
        model = self._make_model_with_norms()
        quantize_model(model, "fp8")
        assert isinstance(model.embed, nn.Embedding)

    def test_quantizes_all_linears(self):
        from trainer.quantization import quantize_model
        from trainer.quantization.base import QuantizedLinear
        model = self._make_model_with_norms()
        quantize_model(model, "fp8")
        # After quantization, no plain nn.Linear should remain
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                pytest.fail(f"Found un-quantized nn.Linear at '{name}'")

    def test_returns_stats_dict(self):
        from trainer.quantization import quantize_model
        model = self._make_model_with_norms()
        stats = quantize_model(model, "fp8")
        assert "quantized" in stats
        assert "skipped" in stats
        assert isinstance(stats["quantized"], int)
        assert isinstance(stats["skipped"], int)

    def test_quantized_count_correct(self):
        from trainer.quantization import quantize_model
        model = self._make_model_with_norms()
        stats = quantize_model(model, "fp8")
        # fc1 and fc2 should be quantized
        assert stats["quantized"] == 2

    def test_skipped_count_correct(self):
        from trainer.quantization import quantize_model
        model = self._make_model_with_norms()
        stats = quantize_model(model, "fp8")
        # norm and embed should be skipped
        assert stats["skipped"] == 2

    def test_nested_model_recursive(self):
        from trainer.quantization import quantize_model
        from trainer.quantization.base import QuantizedLinear
        model = self._make_nested_model()
        stats = quantize_model(model, "fp8")
        # Both inner.linear and top_linear should be quantized
        assert stats["quantized"] == 2
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                pytest.fail(f"Found un-quantized nn.Linear at '{name}'")

    def test_none_type_returns_zero_stats(self):
        from trainer.quantization import quantize_model
        model = self._make_model_with_norms()
        stats = quantize_model(model, "fp8")
        # Sanity: calling with a real type gives non-zero
        assert stats["quantized"] > 0

    def test_quantize_model_with_compute_dtype(self):
        from trainer.quantization import quantize_model
        model = nn.Sequential(nn.Linear(8, 8))
        stats = quantize_model(model, "fp8", compute_dtype=torch.float16)
        assert stats["quantized"] == 1

    def test_fp8_scaled_via_quantize_model(self):
        from trainer.quantization import quantize_model
        from trainer.quantization.fp8 import LinearFp8
        model = nn.Sequential(nn.Linear(16, 8))
        quantize_model(model, "fp8_scaled")
        q = model[0]
        assert isinstance(q, LinearFp8)
        assert q._weight_scale is not None


# ---------------------------------------------------------------------------
# TestNF4
# ---------------------------------------------------------------------------

class TestNF4:
    """Tests for NF4 quantization — conditionally requires bitsandbytes."""

    def test_is_bnb_available_returns_bool(self):
        from trainer.quantization.bnb import is_bnb_available
        result = is_bnb_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")
    def test_nf4_quantize_replaces_linear(self):
        from trainer.quantization.bnb import quantize_linear_nf4, LinearNf4
        lin = nn.Linear(32, 16)
        q = quantize_linear_nf4(lin)
        assert isinstance(q, LinearNf4)

    @pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")
    def test_nf4_has_quantized_weight(self):
        from trainer.quantization.bnb import quantize_linear_nf4
        lin = nn.Linear(32, 16)
        q = quantize_linear_nf4(lin)
        assert q._quantized_weight is not None
        assert q._quant_state is not None

    @pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")
    def test_nf4_via_quantize_model(self):
        from trainer.quantization import quantize_model
        from trainer.quantization.bnb import LinearNf4
        model = nn.Sequential(nn.Linear(64, 32))
        stats = quantize_model(model, "nf4")
        assert stats["quantized"] == 1
        assert isinstance(model[0], LinearNf4)

    @pytest.mark.skipif(_bnb_available(), reason="bitsandbytes IS installed — testing stub path")
    def test_nf4_raises_import_error_without_bnb(self):
        """When bnb is absent, calling the nf4 factory raises ImportError."""
        from trainer.quantization import get_quantizer, _QUANTIZERS, _register_quantizers
        # Force re-registration to ensure stubs are loaded
        _QUANTIZERS.clear()
        _register_quantizers()
        factory = _QUANTIZERS["nf4"]
        with pytest.raises(ImportError, match="bitsandbytes"):
            factory()

    @pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")
    def test_nf4_bias_preserved(self):
        from trainer.quantization.bnb import quantize_linear_nf4
        lin = nn.Linear(32, 16, bias=True)
        q = quantize_linear_nf4(lin)
        assert q.bias is not None
        assert q.bias.shape == (16,)

    @pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")
    def test_int8_quantize_replaces_linear(self):
        from trainer.quantization.bnb import quantize_linear_int8, LinearInt8
        lin = nn.Linear(32, 16)
        q = quantize_linear_int8(lin)
        assert isinstance(q, LinearInt8)
