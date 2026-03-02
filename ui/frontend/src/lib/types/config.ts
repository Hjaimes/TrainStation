// Typed TypeScript interfaces mirroring trainer/config/schema.py Pydantic models.

export interface ModelConfig {
	architecture: string;
	base_model_path: string;
	vae_path: string | null;
	dtype: string;
	vae_dtype: string;
	quantization: string | null;
	attn_mode: string;
	split_attn: boolean;
	gradient_checkpointing: boolean;
	compile_model: boolean;
	block_swap_count: number;
	activation_offloading: boolean;
	weight_bouncing: boolean;
	model_kwargs: Record<string, unknown>;
}

export interface TrainingConfig {
	method: string;
	epochs: number;
	max_steps: number | null;
	batch_size: number;
	gradient_accumulation_steps: number;
	mixed_precision: string;
	seed: number | null;
	max_grad_norm: number;
	noise_offset: number;
	min_timestep: number;
	max_timestep: number;
	timestep_sampling: string;
	discrete_flow_shift: number;
	sigmoid_scale: number;
	logit_mean: number;
	logit_std: number;
	mode_scale: number;
	weighting_scheme: string;
	snr_gamma: number;
	p2_gamma: number;
	zero_terminal_snr: boolean;
	loss_type: string;
	huber_delta: number;
	guidance_scale: number;
	ema_enabled: boolean;
	ema_decay: number;
	ema_device: string;
	resume_from: string | null;
	noise_offset_type: string;
	dynamic_timestep_shift: boolean;
	shift_base: number;
	shift_max: number;
	progressive_timesteps: boolean;
	progressive_warmup_steps: number;
	stochastic_rounding: boolean;
	fused_backward: boolean;
	train_text_encoder: boolean;
	text_encoder_lr: number | null;
	text_encoder_gradient_checkpointing: boolean;
}

export interface OptimizerConfig {
	optimizer_type: string;
	learning_rate: number;
	weight_decay: number;
	scheduler_type: string;
	warmup_steps: number;
	warmup_ratio: number;
	min_lr_ratio: number;
	lr_scaling: string;
	optimizer_kwargs: Record<string, unknown>;
	component_lr_overrides: Record<string, number> | null;
}

export interface NetworkConfig {
	network_type: string;
	rank: number;
	alpha: number;
	dropout: number | null;
	rank_dropout: number | null;
	module_dropout: number | null;
	network_args: Record<string, unknown>;
	scale_weight_norms: number | null;
	loraplus_lr_ratio: number | null;
	network_weights: string | null;
	exclude_patterns: string[];
	include_patterns: string[];
	save_dtype: string | null;
	use_dora: boolean;
	block_lr_multipliers: number[] | null;
}

export interface DatasetEntry {
	path: string;
	caption_extension: string;
	repeats: number;
	weight: number;
	is_video: boolean;
	num_frames: number;
	frame_extraction: string;
}

export interface DataConfig {
	dataset_config_path: string | null;
	datasets: DatasetEntry[];
	cache_latents: boolean;
	cache_latents_to_disk: boolean;
	cache_text_encoder_outputs: boolean;
	num_workers: number;
	persistent_workers: boolean;
	resolution: number;
	enable_bucket: boolean;
	bucket_min_resolution: number;
	bucket_max_resolution: number;
	flip_aug: boolean;
	crop_jitter: number;
	shuffle_tags: boolean;
	keep_tags_count: number;
	token_dropout_rate: number;
	caption_delimiter: string;
	masked_training: boolean;
	mask_weight: number;
	unmasked_probability: number;
	normalize_masked_area_loss: boolean;
	reg_data_path: string | null;
	prior_loss_weight: number;
}

export interface SamplingConfig {
	enabled: boolean;
	prompts: string[];
	prompts_file: string | null;
	sample_every_n_steps: number | null;
	sample_every_n_epochs: number | null;
	sample_at_first: boolean;
	width: number;
	height: number;
	num_frames: number;
	num_inference_steps: number;
	guidance_scale: number;
	seed: number | null;
}

export interface SavingConfig {
	output_dir: string;
	output_name: string;
	save_every_n_steps: number | null;
	save_every_n_epochs: number | null;
	max_keep_ckpts: number | null;
}

export interface LoggingConfig {
	logging_dir: string | null;
	log_with: string | null;
	log_prefix: string | null;
	vram_profiling: boolean;
}

export interface ValidationConfig {
	enabled: boolean;
	data_path: string | null;
	interval_steps: number;
	num_steps: number;
	fixed_timestep: number;
}

export interface TrainConfig {
	version: number;
	model: ModelConfig;
	training: TrainingConfig;
	optimizer: OptimizerConfig;
	data: DataConfig;
	saving: SavingConfig;
	network: NetworkConfig | null;
	sampling: SamplingConfig;
	logging: LoggingConfig;
	validation: ValidationConfig;
}
