"""All configuration models. Pydantic v2 with strict validation."""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    """Model architecture and loading configuration."""
    architecture: str                            # "wan", "hunyuan_video", "flux_2", etc.
    base_model_path: str                         # Path to transformer/DiT/UNet weights
    vae_path: str | None = None
    dtype: str = "bf16"                          # bf16, fp16, fp32
    vae_dtype: str = "bf16"
    quantization: str | None = None              # None, "nf4", "int8", "fp8", "fp8_scaled"
    attn_mode: str = "sdpa"                      # sdpa, flash, xformers
    split_attn: bool = False
    gradient_checkpointing: bool = True
    compile_model: bool = False
    block_swap_count: int = 0
    activation_offloading: bool = False
    weight_bouncing: bool = False
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid", "validate_assignment": True}


class TrainingConfig(BaseModel):
    """Training loop parameters."""
    method: str = "lora"                         # "lora", "full_finetune"
    epochs: int = 1
    max_steps: int | None = None                 # If set, overrides epochs
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"
    seed: int | None = None
    max_grad_norm: float = 1.0
    noise_offset: float = 0.0
    min_timestep: float = 0.0
    max_timestep: float = 1.0
    timestep_sampling: str = "uniform"           # uniform, sigmoid, logit_normal
    discrete_flow_shift: float = 1.0
    sigmoid_scale: float = 1.0
    logit_mean: float = 0.0
    logit_std: float = 1.0
    mode_scale: float = 1.29
    weighting_scheme: str = "none"               # none, min_snr_gamma, debiased, p2
    snr_gamma: float = 5.0                       # gamma for min_snr_gamma weighting
    p2_gamma: float = 1.0                        # gamma exponent for p2 weighting
    zero_terminal_snr: bool = False              # rescale noise schedule so final timestep has SNR=0
    loss_type: str = "mse"                       # mse, l1, mae, huber
    huber_delta: float = 1.0                     # delta for huber loss
    guidance_scale: float = 1.0
    ema_enabled: bool = False
    ema_decay: float = 0.9999
    ema_device: str = "cpu"                      # cpu or cuda - cpu avoids doubling VRAM
    resume_from: str | None = None
    noise_offset_type: str = "simple"            # "simple" or "generalized" (psi(t) = offset * sqrt(t))
    dynamic_timestep_shift: bool = False         # Enable resolution-dependent timestep shifting (Flux/SD3 style)
    shift_base: float = 0.5                      # Base shift mu at base_seq_len
    shift_max: float = 1.15                      # Max shift mu at max_seq_len
    progressive_timesteps: bool = False          # Blend from uniform toward target distribution during warmup
    progressive_warmup_steps: int = 1000         # Steps to linearly blend from uniform to full distribution
    stochastic_rounding: bool = False            # Apply stochastic rounding when casting fp32 optimizer states to bf16
    fused_backward: bool = False                 # Step each param during backward to avoid storing all grads (saves ~25-40% VRAM, single-GPU only)
    train_text_encoder: bool = False             # Enable text encoder training (raw captions loaded alongside cached latents)
    text_encoder_lr: float | None = None         # Separate LR for text encoder (None = use base LR)
    text_encoder_gradient_checkpointing: bool = True  # Gradient checkpointing for text encoder

    model_config = {"extra": "forbid", "validate_assignment": True}


class OptimizerConfig(BaseModel):
    """Optimizer and learning rate configuration."""
    optimizer_type: str = "adamw"                # adamw, adamw8bit, adafactor, prodigy, lion, came, schedule_free_adamw
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    scheduler_type: str = "cosine"               # cosine, constant, linear, constant_with_warmup, exponential, inverse_sqrt
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    min_lr_ratio: float = 0.0
    lr_scaling: str = "none"                     # none, linear, sqrt - scales lr by effective batch size
    optimizer_kwargs: dict[str, Any] = Field(default_factory=dict)
    component_lr_overrides: dict[str, float] | None = None  # e.g. {"text_encoder": 1e-5, "norm": 5e-5}

    model_config = {"extra": "forbid", "validate_assignment": True}


class NetworkConfig(BaseModel):
    """LoRA/LoHa/LoKr/DoRA network configuration."""
    network_type: str = "lora"                   # lora, loha, lokr, dora
    rank: int = 16
    alpha: float = 16.0
    dropout: float | None = None
    rank_dropout: float | None = None
    module_dropout: float | None = None
    network_args: dict[str, Any] = Field(default_factory=dict)
    scale_weight_norms: float | None = None
    loraplus_lr_ratio: float | None = None
    network_weights: str | None = None           # Path to pre-trained network weights
    exclude_patterns: list[str] = Field(default_factory=list)
    include_patterns: list[str] = Field(default_factory=list)
    save_dtype: str | None = None
    use_dora: bool = False                       # Weight-Decomposed LoRA; overrides network_type="lora" to use DoRA
    block_lr_multipliers: list[float] | None = None  # Per-block LR multipliers; index maps to block number

    model_config = {"extra": "forbid", "validate_assignment": True}


class DatasetEntry(BaseModel):
    """One dataset within the training config (inline mode)."""
    path: str
    caption_extension: str = ".txt"
    repeats: int = 10
    weight: float = 1.0
    is_video: bool = False
    num_frames: int = 1
    frame_extraction: str = "uniform"            # uniform, head

    model_config = {"extra": "forbid"}


class DataConfig(BaseModel):
    """Dataset and data loading configuration."""
    dataset_config_path: str | None = None       # Path to TOML (Musubi compat)
    datasets: list[DatasetEntry] = Field(default_factory=list)
    cache_latents: bool = True
    cache_latents_to_disk: bool = False
    cache_text_encoder_outputs: bool = True
    num_workers: int = 2
    persistent_workers: bool = True
    resolution: int = 512
    enable_bucket: bool = True
    bucket_min_resolution: int = 256
    bucket_max_resolution: int = 2048
    flip_aug: bool = False
    crop_jitter: int = 0                             # Pixel-space crop jitter radius (0 = disabled)
    shuffle_tags: bool = False                       # Randomly shuffle comma-separated tags each step
    keep_tags_count: int = 0                         # Number of leading tags to keep fixed during shuffle/dropout
    token_dropout_rate: float = 0.0                  # Fraction of non-fixed tags to drop each step
    caption_delimiter: str = ","                     # Delimiter used to split caption into tags
    masked_training: bool = False                    # Enable masked loss computation
    mask_weight: float = 1.0                         # Weight for masked regions (>1 = increase emphasis, <1 = reduce)
    unmasked_probability: float = 0.0               # Probability of ignoring the mask entirely for a given sample
    normalize_masked_area_loss: bool = True          # Normalize loss by mask area to keep scale stable
    reg_data_path: str | None = None                 # Path to regularization data TOML (prior preservation)
    prior_loss_weight: float = 1.0                   # Weight for regularization loss (prior preservation)

    model_config = {"extra": "forbid", "validate_assignment": True}


class SamplingConfig(BaseModel):
    """Sample generation during training."""
    enabled: bool = False
    prompts: list[str] = Field(default_factory=list)
    prompts_file: str | None = None
    sample_every_n_steps: int | None = None
    sample_every_n_epochs: int | None = None
    sample_at_first: bool = False
    width: int = 512
    height: int = 512
    num_frames: int = 1
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seed: int | None = 42

    model_config = {"extra": "forbid", "validate_assignment": True}


class SavingConfig(BaseModel):
    """Output and checkpoint saving configuration."""
    output_dir: str = "./output"
    output_name: str = "trained"
    save_every_n_steps: int | None = None
    save_every_n_epochs: int | None = 1
    max_keep_ckpts: int | None = None

    model_config = {"extra": "forbid", "validate_assignment": True}


class LoggingConfig(BaseModel):
    """Logging and experiment tracking."""
    logging_dir: str | None = None
    log_with: str | None = None                  # "tensorboard", "wandb"
    log_prefix: str | None = None
    vram_profiling: bool = False                 # Attach VRAMProfilerCallback during training

    model_config = {"extra": "forbid", "validate_assignment": True}


class ValidationConfig(BaseModel):
    """Validation loss configuration."""
    enabled: bool = False
    data_path: str | None = None                 # Path to validation data TOML
    interval_steps: int = 500                    # Run validation every N steps
    num_steps: int = 10                          # Number of validation batches to average
    fixed_timestep: float = 0.5                  # Reserved: fixed timestep for reproducibility

    model_config = {"extra": "forbid", "validate_assignment": True}


class TrainConfig(BaseModel):
    """Top-level composite config. The single source of truth."""
    version: int = 1
    model: ModelConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    saving: SavingConfig = Field(default_factory=SavingConfig)
    network: NetworkConfig | None = None
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    model_config = {"extra": "forbid", "validate_assignment": True}

    @model_validator(mode="after")
    def cross_validate(self) -> "TrainConfig":
        errors: list[str] = []
        if self.training.method == "lora" and self.network is None:
            errors.append("training.method='lora' requires a [network] section")
        if self.training.method == "full_finetune" and self.network is not None:
            errors.append("training.method='full_finetune' should not have a [network] section")
        if self.network is not None and self.network.rank <= 0:
            errors.append(f"network.rank must be > 0, got {self.network.rank}")
        if not self.model.base_model_path:
            errors.append("model.base_model_path must not be empty")
        if self.data.dataset_config_path is None and len(self.data.datasets) == 0:
            errors.append("Either data.dataset_config_path or data.datasets must be provided")
        if errors:
            raise ValueError("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
        return self

    def freeze(self) -> "TrainConfig":
        """Return an immutable deep copy for the training runtime."""
        return self.model_copy(deep=True)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        return cls.model_validate(data)
