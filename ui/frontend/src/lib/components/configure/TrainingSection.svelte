<script lang="ts">
    import { configState } from '$lib/stores/config';
    import { FormField, SelectInput, NumberInput, ToggleInput } from '$lib/components/form';
    import type { TrainingConfig, OptimizerConfig } from '$lib/types/config';

    let state = $derived($configState);
    let training = $derived(state.config?.training);
    let optimizer = $derived(state.config?.optimizer);
    let showAdvanced = $state(false);

    let useMaxSteps = $state(false);

    // Derive conditional visibility
    let showSnrGamma = $derived(training?.weighting_scheme === 'min_snr_gamma');
    let showP2Gamma = $derived(training?.weighting_scheme === 'p2');
    let showEmaDecay = $derived(training?.ema_enabled ?? false);
    let showTextEncoderLr = $derived(training?.train_text_encoder ?? false);

    function updateTraining(field: keyof TrainingConfig, value: unknown) {
        configState.updateSection('training', { [field]: value } as Partial<TrainingConfig>);
    }

    function updateOptimizer(field: keyof OptimizerConfig, value: unknown) {
        configState.updateSection('optimizer', { [field]: value } as Partial<OptimizerConfig>);
    }

    function handleStepsModeToggle() {
        useMaxSteps = !useMaxSteps;
        if (!useMaxSteps) {
            updateTraining('max_steps', null);
        }
    }
</script>

<div class="section">
    <h2 class="section-title">Training</h2>

    <div class="fields">
        <FormField label="Training Method" description="Training approach">
            <SelectInput
                value={training?.method ?? 'lora'}
                options={[
                    { value: 'lora', label: 'LoRA' },
                    { value: 'full_finetune', label: 'Full Fine-Tune' },
                ]}
                onchange={(v) => updateTraining('method', v)}
            />
        </FormField>

        <FormField label="Optimizer" description="Optimization algorithm">
            <SelectInput
                value={optimizer?.optimizer_type ?? 'adamw'}
                options={[
                    { value: 'adamw', label: 'AdamW' },
                    { value: 'adamw8bit', label: 'AdamW 8-bit' },
                    { value: 'adafactor', label: 'Adafactor' },
                    { value: 'prodigy', label: 'Prodigy' },
                    { value: 'lion', label: 'Lion' },
                    { value: 'came', label: 'CAME' },
                    { value: 'schedule_free_adamw', label: 'Schedule-Free AdamW' },
                ]}
                onchange={(v) => updateOptimizer('optimizer_type', v)}
            />
        </FormField>

        <FormField label="Learning Rate" description="Base learning rate">
            <NumberInput
                value={optimizer?.learning_rate ?? 1e-4}
                step="any"
                onchange={(v) => updateOptimizer('learning_rate', v)}
            />
        </FormField>

        <FormField label="LR Scheduler" description="Learning rate schedule">
            <SelectInput
                value={optimizer?.scheduler_type ?? 'cosine'}
                options={[
                    { value: 'cosine', label: 'Cosine' },
                    { value: 'constant', label: 'Constant' },
                    { value: 'linear', label: 'Linear' },
                    { value: 'constant_with_warmup', label: 'Constant with Warmup' },
                    { value: 'exponential', label: 'Exponential' },
                    { value: 'inverse_sqrt', label: 'Inverse Sqrt' },
                ]}
                onchange={(v) => updateOptimizer('scheduler_type', v)}
            />
        </FormField>

        <div class="steps-mode">
            <div class="steps-toggle">
                <button
                    type="button"
                    class="mode-btn"
                    class:active={!useMaxSteps}
                    onclick={() => { if (useMaxSteps) handleStepsModeToggle(); }}
                >Epochs</button>
                <button
                    type="button"
                    class="mode-btn"
                    class:active={useMaxSteps}
                    onclick={() => { if (!useMaxSteps) handleStepsModeToggle(); }}
                >Max Steps</button>
            </div>
            {#if useMaxSteps}
                <FormField label="Max Steps" description="Overrides epochs when set">
                    <NumberInput
                        value={training?.max_steps ?? 1000}
                        min={1}
                        onchange={(v) => updateTraining('max_steps', v)}
                    />
                </FormField>
            {:else}
                <FormField label="Epochs" description="Number of training epochs">
                    <NumberInput
                        value={training?.epochs ?? 10}
                        min={1}
                        onchange={(v) => updateTraining('epochs', v)}
                    />
                </FormField>
            {/if}
        </div>

        <FormField label="Gradient Accumulation" description="Steps to accumulate before update">
            <NumberInput
                value={training?.gradient_accumulation_steps ?? 1}
                min={1}
                onchange={(v) => updateTraining('gradient_accumulation_steps', v)}
            />
        </FormField>
    </div>

    <button class="toggle-advanced" type="button" onclick={() => showAdvanced = !showAdvanced}>
        {showAdvanced ? '\u25BE Hide' : '\u25B8 Show'} Advanced Settings
    </button>

    {#if showAdvanced}
    <div class="fields advanced-fields">
        <FormField label="Seed" description="Random seed (empty for random)">
            <NumberInput
                value={training?.seed ?? 0}
                onchange={(v) => updateTraining('seed', v || null)}
            />
        </FormField>

        <FormField label="Warmup Steps" description="LR warmup step count">
            <NumberInput
                value={optimizer?.warmup_steps ?? 0}
                min={0}
                onchange={(v) => updateOptimizer('warmup_steps', v)}
            />
        </FormField>

        <FormField label="Min LR Ratio" description="Minimum LR as fraction of base LR">
            <NumberInput
                value={optimizer?.min_lr_ratio ?? 0.0}
                min={0}
                max={1}
                step={0.01}
                onchange={(v) => updateOptimizer('min_lr_ratio', v)}
            />
        </FormField>

        <FormField label="Weight Decay" description="L2 regularization weight">
            <NumberInput
                value={optimizer?.weight_decay ?? 0.01}
                min={0}
                step={0.001}
                onchange={(v) => updateOptimizer('weight_decay', v)}
            />
        </FormField>

        <FormField label="Max Grad Norm" description="Gradient clipping threshold">
            <NumberInput
                value={training?.max_grad_norm ?? 1.0}
                min={0}
                step={0.1}
                onchange={(v) => updateTraining('max_grad_norm', v)}
            />
        </FormField>

        <FormField label="Mixed Precision" description="Training precision mode">
            <SelectInput
                value={training?.mixed_precision ?? 'bf16'}
                options={[
                    { value: 'bf16', label: 'BF16' },
                    { value: 'fp16', label: 'FP16' },
                    { value: 'fp32', label: 'FP32' },
                ]}
                onchange={(v) => updateTraining('mixed_precision', v)}
            />
        </FormField>

        <FormField label="Noise Offset" description="Noise offset for improved contrast">
            <NumberInput
                value={training?.noise_offset ?? 0}
                min={0}
                step={0.001}
                onchange={(v) => updateTraining('noise_offset', v)}
            />
        </FormField>

        <FormField label="Noise Offset Type" description="How noise offset is applied">
            <SelectInput
                value={training?.noise_offset_type ?? 'simple'}
                options={[
                    { value: 'simple', label: 'Simple' },
                    { value: 'generalized', label: 'Generalized' },
                ]}
                onchange={(v) => updateTraining('noise_offset_type', v)}
            />
        </FormField>

        <FormField label="Timestep Sampling" description="How timesteps are sampled">
            <SelectInput
                value={training?.timestep_sampling ?? 'uniform'}
                options={[
                    { value: 'uniform', label: 'Uniform' },
                    { value: 'sigmoid', label: 'Sigmoid' },
                    { value: 'logit_normal', label: 'Logit Normal' },
                ]}
                onchange={(v) => updateTraining('timestep_sampling', v)}
            />
        </FormField>

        <FormField label="Loss Type" description="Loss function">
            <SelectInput
                value={training?.loss_type ?? 'mse'}
                options={[
                    { value: 'mse', label: 'MSE' },
                    { value: 'l1', label: 'L1' },
                    { value: 'mae', label: 'MAE' },
                    { value: 'huber', label: 'Huber' },
                ]}
                onchange={(v) => updateTraining('loss_type', v)}
            />
        </FormField>

        <FormField label="Weighting Scheme" description="Loss weighting strategy">
            <SelectInput
                value={training?.weighting_scheme ?? 'none'}
                options={[
                    { value: 'none', label: 'None' },
                    { value: 'min_snr_gamma', label: 'Min SNR Gamma' },
                    { value: 'debiased', label: 'Debiased' },
                    { value: 'p2', label: 'P2' },
                ]}
                onchange={(v) => updateTraining('weighting_scheme', v)}
            />
        </FormField>

        {#if showSnrGamma}
        <FormField label="SNR Gamma" description="Min SNR gamma value">
            <NumberInput
                value={training?.snr_gamma ?? 5.0}
                min={0}
                step={0.1}
                onchange={(v) => updateTraining('snr_gamma', v)}
            />
        </FormField>
        {/if}

        {#if showP2Gamma}
        <FormField label="P2 Gamma" description="P2 weighting gamma">
            <NumberInput
                value={training?.p2_gamma ?? 1.0}
                min={0}
                step={0.1}
                onchange={(v) => updateTraining('p2_gamma', v)}
            />
        </FormField>
        {/if}

        <FormField label="Zero Terminal SNR" description="Force zero terminal SNR">
            <ToggleInput
                value={training?.zero_terminal_snr ?? false}
                onchange={(v) => updateTraining('zero_terminal_snr', v)}
            />
        </FormField>

        <FormField label="Guidance Scale" description="Classifier-free guidance scale">
            <NumberInput
                value={training?.guidance_scale ?? 1.0}
                min={0}
                step={0.1}
                onchange={(v) => updateTraining('guidance_scale', v)}
            />
        </FormField>

        <FormField label="EMA Enabled" description="Exponential moving average of weights">
            <ToggleInput
                value={training?.ema_enabled ?? false}
                onchange={(v) => updateTraining('ema_enabled', v)}
            />
        </FormField>

        {#if showEmaDecay}
        <FormField label="EMA Decay" description="EMA decay rate">
            <NumberInput
                value={training?.ema_decay ?? 0.9999}
                min={0}
                max={1}
                step={0.0001}
                onchange={(v) => updateTraining('ema_decay', v)}
            />
        </FormField>
        {/if}

        <FormField label="Stochastic Rounding" description="Stochastic rounding for BF16 training">
            <ToggleInput
                value={training?.stochastic_rounding ?? false}
                onchange={(v) => updateTraining('stochastic_rounding', v)}
            />
        </FormField>

        <FormField label="Fused Backward" description="Fuse optimizer step with backward pass">
            <ToggleInput
                value={training?.fused_backward ?? false}
                onchange={(v) => updateTraining('fused_backward', v)}
            />
        </FormField>

        <FormField label="LR Scaling" description="Scale LR by batch size">
            <SelectInput
                value={optimizer?.lr_scaling ?? 'none'}
                options={[
                    { value: 'none', label: 'None' },
                    { value: 'linear', label: 'Linear' },
                    { value: 'sqrt', label: 'Square Root' },
                ]}
                onchange={(v) => updateOptimizer('lr_scaling', v)}
            />
        </FormField>

        <FormField label="Train Text Encoder" description="Also train the text encoder">
            <ToggleInput
                value={training?.train_text_encoder ?? false}
                onchange={(v) => updateTraining('train_text_encoder', v)}
            />
        </FormField>

        {#if showTextEncoderLr}
        <FormField label="Text Encoder LR" description="Separate LR for text encoder">
            <NumberInput
                value={training?.text_encoder_lr ?? 1e-5}
                step="any"
                onchange={(v) => updateTraining('text_encoder_lr', v || null)}
            />
        </FormField>
        {/if}
    </div>
    {/if}
</div>

<style>
    .section {
        display: flex;
        flex-direction: column;
        gap: 16px;
    }
    .section-title {
        font-family: var(--font-mono);
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-primary);
        margin: 0;
    }
    .fields {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
    }
    .steps-mode {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .steps-toggle {
        display: flex;
        gap: 0;
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius);
        overflow: hidden;
        width: fit-content;
    }
    .mode-btn {
        padding: 4px 12px !important;
        font-family: var(--font-mono);
        font-size: 11px;
        background: var(--bg-secondary) !important;
        border: none !important;
        border-right: 1px solid var(--border-subtle) !important;
        color: var(--text-muted);
        cursor: pointer;
    }
    .mode-btn:last-child {
        border-right: none !important;
    }
    .mode-btn.active {
        background: var(--accent-dim) !important;
        color: var(--accent);
        font-weight: 600;
    }
    .toggle-advanced {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: none !important;
        border: none !important;
        padding: 4px 0 !important;
        color: var(--text-muted);
        font-family: var(--font-mono);
        font-size: 11px;
        cursor: pointer;
        text-transform: none;
        letter-spacing: normal;
    }
    .toggle-advanced:hover {
        color: var(--accent);
    }
    .advanced-fields {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        padding-top: 4px;
        border-top: 1px solid var(--border-subtle);
    }
</style>
