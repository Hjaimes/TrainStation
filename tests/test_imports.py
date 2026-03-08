"""Canary imports: verify every Phase 1 module imports without error."""


def test_import_trainer():
    import trainer


def test_import_config_schema():
    from trainer.config.schema import (
        TrainConfig, ModelConfig, TrainingConfig, OptimizerConfig,
        NetworkConfig, DataConfig, DatasetEntry, SavingConfig,
        SamplingConfig, LoggingConfig,
    )


def test_import_config_io():
    from trainer.config.io import load_config, save_config, apply_overrides, load_config_from_dict


def test_import_config_validation():
    from trainer.config.validation import validate_config, ValidationResult, ValidationIssue


def test_import_config_init():
    from trainer.config import TrainConfig, validate_config, load_config


def test_import_errors():
    from trainer.errors import TrainerError, ConfigError, ModelLoadError


def test_import_registry():
    from trainer.registry import register_model, get_model_strategy, list_models


def test_import_callbacks():
    from trainer.callbacks import TrainingCallback, CLIProgressCallback, StepMetrics


def test_import_events():
    from trainer.events import (
        TrainingEvent, TrainingStartedEvent, StepEvent, EpochEvent,
        SampleEvent, CheckpointEvent, LogEvent, ErrorEvent, TrainingCompleteEvent,
        StopCommand, PauseCommand, ResumeCommand, SampleCommand, SaveCommand,
    )


def test_import_arch_base():
    from trainer.arch.base import ModelStrategy, ModelComponents, TrainStepOutput


def test_import_util_dtype():
    from trainer.util import resolve_dtype, dtype_to_str


def test_import_util_timer():
    from trainer.util import TrainingTimer


# Phase 3: Wan components
def test_import_wan_strategy():
    from trainer.arch.wan.strategy import WanStrategy


def test_import_wan_configs():
    from trainer.arch.wan.components.configs import WAN_CONFIGS


def test_import_wan_model():
    from trainer.arch.wan.components.model import WanModel, load_wan_model


def test_import_wan_attention():
    from trainer.arch.wan.components.attention import flash_attention


def test_import_wan_vae():
    from trainer.arch.wan.components.vae import WanVAE


def test_import_wan_t5():
    from trainer.arch.wan.components.t5 import T5EncoderModel


def test_import_wan_clip():
    from trainer.arch.wan.components.clip import CLIPModel


def test_import_wan_utils():
    from trainer.arch.wan.components.utils import (
        ModelOffloader, MemoryEfficientSafeOpen,
        load_safetensors_with_lora_and_fp8, load_safetensors,
        clean_memory_on_device,
    )


# Phase 3: Data pipeline
def test_import_data_dataset():
    from trainer.data.dataset import (
        ItemInfo, BucketBatchManager, CachedDataset, CachedDatasetGroup,
    )


def test_import_data_loader():
    from trainer.data.loader import create_dataloader


def test_import_data_toml():
    from trainer.data.toml_config import parse_toml_config


def test_import_data_caching():
    from trainer.data.loader import check_cache_exists, log_caching_instructions


def test_import_data_init():
    from trainer.data import (
        ItemInfo, CachedDataset, CachedDatasetGroup,
        create_dataloader, parse_toml_config,
        check_cache_exists, log_caching_instructions,
    )


def test_import_data_text_processing():
    from trainer.data.text_processing import shuffle_tags, apply_token_dropout, process_caption


def test_import_data_augmentations():
    from trainer.data.augmentations import apply_crop_jitter, apply_random_flip


def test_import_callbacks_vram():
    from trainer.callbacks import VRAMProfilerCallback


# Phase 4: UI / subprocess
def test_import_ui_runner():
    from ui.runner import SubprocessTrainingRunner, _training_worker


def test_import_ui_binding():
    from ui.binding import ConfigBinder


def test_import_ui_server():
    from ui.server import app, dataclass_to_dict


def test_import_ui_routes_training():
    from ui.routes.training import router


def test_import_ui_routes_config():
    from ui.routes.config import router


def test_import_ui_routes_models():
    from ui.routes.models import router


# Phase 5: Tier 1 architectures
def test_import_zimage_strategy():
    from trainer.arch.zimage.strategy import ZImageStrategy


def test_import_zimage_configs():
    from trainer.arch.zimage.components.configs import ZIMAGE_CONFIGS


def test_import_flux2_strategy():
    from trainer.arch.flux_2.strategy import Flux2Strategy


def test_import_flux2_configs():
    from trainer.arch.flux_2.components.configs import FLUX2_CONFIGS


def test_import_qwen_image_strategy():
    from trainer.arch.qwen_image.strategy import QwenImageStrategy


def test_import_qwen_image_configs():
    from trainer.arch.qwen_image.components.configs import QWEN_IMAGE_CONFIGS


def test_import_flux_kontext_strategy():
    from trainer.arch.flux_kontext.strategy import FluxKontextStrategy


def test_import_flux_kontext_configs():
    from trainer.arch.flux_kontext.components.configs import FLUX_KONTEXT_CONFIGS


def test_import_flux_kontext_model():
    from trainer.arch.flux_kontext.components.model import (
        FluxKontextModel, detect_flux_kontext_weight_dtype, load_flux_kontext_model,
    )


def test_import_flux_kontext_utils():
    from trainer.arch.flux_kontext.components.utils import (
        prepare_img_ids, prepare_txt_ids, pack_latents, unpack_latents,
    )


# Phase 5: Tier 2 architectures
def test_import_framepack_strategy():
    from trainer.arch.framepack.strategy import FramePackStrategy


def test_import_framepack_configs():
    from trainer.arch.framepack.components.configs import FRAMEPACK_CONFIGS


def test_import_kandinsky5_strategy():
    from trainer.arch.kandinsky5.strategy import Kandinsky5Strategy


def test_import_kandinsky5_configs():
    from trainer.arch.kandinsky5.components.configs import TASK_CONFIGS


# Phase 5: Tier 3 architectures
def test_import_hunyuan_video_strategy():
    from trainer.arch.hunyuan_video.strategy import HunyuanVideoStrategy


def test_import_hunyuan_video_configs():
    from trainer.arch.hunyuan_video.components.configs import HunyuanVideoConfig


def test_import_hv_1_5_strategy():
    from trainer.arch.hunyuan_video_1_5.strategy import HunyuanVideo15Strategy


def test_import_hv_1_5_configs():
    from trainer.arch.hunyuan_video_1_5.components.configs import HV15ModelConfig


# Phase 5b: SD3 architecture
def test_import_sd3_strategy():
    from trainer.arch.sd3.strategy import SD3Strategy


def test_import_sd3_configs():
    from trainer.arch.sd3.components.configs import SD3_CONFIGS


def test_import_sd3_model():
    from trainer.arch.sd3.components.model import SD3Transformer2DModel, load_sd3_model


def test_import_sd3_layers():
    from trainer.arch.sd3.components.layers import (
        AdaLayerNormZero, AdaLayerNormZeroSingle, AdaLayerNormContinuous, FeedForward,
    )


def test_import_sd3_embeddings():
    from trainer.arch.sd3.components.embeddings import (
        PatchEmbed, Timesteps, TimestepEmbedding, CombinedTimestepTextProjEmbeddings,
    )


def test_import_sd3_blocks():
    from trainer.arch.sd3.components.blocks import JointTransformerBlock, SD3SingleTransformerBlock


# Phase 5b: SDXL architecture
def test_import_sdxl_strategy():
    from trainer.arch.sdxl.strategy import SDXLStrategy


def test_import_sdxl_configs():
    from trainer.arch.sdxl.components.configs import SDXL_CONFIGS


def test_import_sdxl_model():
    from trainer.arch.sdxl.components.model import load_sdxl_unet


def test_import_sdxl_utils():
    from trainer.arch.sdxl.components.utils import (
        compute_alphas_cumprod, build_time_ids, get_velocity,
    )


# Phase 5b: Flux 1 architecture
def test_import_flux1_strategy():
    from trainer.arch.flux_1.strategy import Flux1Strategy


def test_import_flux1_configs():
    from trainer.arch.flux_1.components.configs import FLUX1_CONFIGS


def test_import_flux1_model():
    from trainer.arch.flux_1.components.model import Flux1Transformer, load_flux1_model


def test_import_flux1_utils():
    from trainer.arch.flux_1.components.utils import (
        pack_latents, unpack_latents, prepare_img_ids, prepare_txt_ids,
    )


# Phase 2: Core training utilities
def test_import_loss():
    from trainer.loss import get_loss_fn, get_unreduced_loss_fn, compute_loss


def test_import_loss_weighting():
    from trainer.loss_weighting import get_weight_fn, compute_snr_flow_matching


def test_import_ema():
    from trainer.ema import EMATracker


def test_import_quantization():
    from trainer.quantization import quantize_model, get_quantizer


def test_import_quantization_base():
    from trainer.quantization.base import QuantizedLinear


def test_import_quantization_fp8():
    from trainer.quantization.fp8 import LinearFp8, quantize_linear_fp8


def test_import_quantization_bnb():
    from trainer.quantization.bnb import is_bnb_available


def test_import_quantization_utils():
    from trainer.quantization.utils import replace_linear_layers


# Comprehensive feature modules (Tasks 3-28)

def test_import_data_mask_utils():
    from trainer.data.mask_utils import load_mask, normalize_mask


def test_import_data_reg_loader():
    from trainer.data.loader import RegDataIterator


def test_import_networks_dora():
    from trainer.networks.dora import DoRAModule


def test_import_networks_get_module_class_dora():
    from trainer.networks import get_module_class
    cls = get_module_class("dora")
    assert cls.__name__ == "DoRAModule"


def test_import_training_validation():
    from trainer.training.validation import ValidationRunner


def test_import_training_fused_backward():
    from trainer.training.fused_backward import FusedBackwardManager


def test_import_util_stochastic_rounding():
    from trainer.util.stochastic_rounding import (
        copy_stochastic_, register_stochastic_rounding_hook,
    )


def test_import_util_activation_offload():
    from trainer.util.activation_offload import ActivationOffloadContext


def test_import_util_weight_bouncing():
    from trainer.util.weight_bouncing import (
        BouncingLinear, apply_weight_bouncing,
    )


def test_import_validation_config():
    from trainer.config.schema import ValidationConfig


# Phase 6: UI modules
def test_import_ui_presets():
    from ui.presets import PresetManager


def test_import_ui_queue():
    from ui.queue import QueueManager


def test_import_ui_routes_presets():
    from ui.routes.presets import router


def test_import_ui_routes_queue():
    from ui.routes.queue import router


def test_import_ui_routes_samples():
    from ui.routes.samples import router


def test_import_ui_routes_preflight():
    from ui.routes.preflight import router


def test_import_ui_routes_browse():
    from ui.routes.browse import router


def test_import_util_hf_utils():
    from trainer.util.hf_utils import is_huggingface_id, resolve_path_if_hf, find_safetensors_in_dir


def test_import_adamw_advanced():
    from trainer.adamw_advanced import AdamWAdvanced
