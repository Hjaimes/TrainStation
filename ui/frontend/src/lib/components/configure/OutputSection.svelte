<script lang="ts">
    import { configState } from '$lib/stores/config';
    import { FormField, SelectInput, PathInput, NumberInput, TextInput, ToggleInput } from '$lib/components/form';
    import type { SavingConfig, LoggingConfig, ValidationConfig } from '$lib/types/config';

    let cfg = $derived($configState);
    let saving = $derived(cfg.config?.saving);
    let logging = $derived(cfg.config?.logging);
    let validation = $derived(cfg.config?.validation);
    let showAdvanced = $state(false);

    function updateSaving(field: keyof SavingConfig, value: unknown) {
        configState.updateSection('saving', { [field]: value } as Partial<SavingConfig>);
    }

    function updateLogging(field: keyof LoggingConfig, value: unknown) {
        configState.updateSection('logging', { [field]: value } as Partial<LoggingConfig>);
    }

    function updateValidation(field: keyof ValidationConfig, value: unknown) {
        configState.updateSection('validation', { [field]: value } as Partial<ValidationConfig>);
    }
</script>

<div class="section">
    <h2 class="section-title">Output</h2>

    <div class="fields">
        <FormField label="Output Directory" description="Where to save trained weights">
            <PathInput
                value={saving?.output_dir ?? ''}
                onchange={(v) => updateSaving('output_dir', v)}
            />
        </FormField>

        <FormField label="Output Name" description="Base name for saved files">
            <TextInput
                value={saving?.output_name ?? ''}
                onchange={(v) => updateSaving('output_name', v)}
            />
        </FormField>

        <FormField label="Save Every N Steps" description="Steps between checkpoint saves">
            <NumberInput
                value={saving?.save_every_n_steps ?? 500}
                min={0}
                onchange={(v) => updateSaving('save_every_n_steps', v || null)}
            />
        </FormField>

        <FormField label="Max Keep Checkpoints" description="Maximum checkpoints to retain (0 = all)">
            <NumberInput
                value={saving?.max_keep_ckpts ?? 5}
                min={0}
                onchange={(v) => updateSaving('max_keep_ckpts', v || null)}
            />
        </FormField>
    </div>

    <button class="toggle-advanced" type="button" onclick={() => showAdvanced = !showAdvanced}>
        {showAdvanced ? '\u25BE Hide' : '\u25B8 Show'} Advanced Settings
    </button>

    {#if showAdvanced}
    <div class="fields advanced-fields">
        <FormField label="Save Every N Epochs" description="Epochs between checkpoint saves">
            <NumberInput
                value={saving?.save_every_n_epochs ?? 0}
                min={0}
                onchange={(v) => updateSaving('save_every_n_epochs', v || null)}
            />
        </FormField>

        <FormField label="Logging Directory" description="Path for training logs">
            <PathInput
                value={logging?.logging_dir ?? ''}
                onchange={(v) => updateLogging('logging_dir', v || null)}
            />
        </FormField>

        <FormField label="Log With" description="Logging backend">
            <SelectInput
                value={logging?.log_with ?? 'none'}
                options={[
                    { value: 'none', label: 'None' },
                    { value: 'tensorboard', label: 'TensorBoard' },
                    { value: 'wandb', label: 'Weights & Biases' },
                ]}
                onchange={(v) => updateLogging('log_with', v === 'none' ? null : v)}
            />
        </FormField>

        <FormField label="VRAM Profiling" description="Log GPU memory usage">
            <ToggleInput
                value={logging?.vram_profiling ?? false}
                onchange={(v) => updateLogging('vram_profiling', v)}
            />
        </FormField>

        <FormField label="Validation Enabled" description="Run validation during training">
            <ToggleInput
                value={validation?.enabled ?? false}
                onchange={(v) => updateValidation('enabled', v)}
            />
        </FormField>

        {#if validation?.enabled}
        <FormField label="Validation Data Path" description="Path to validation dataset">
            <PathInput
                value={validation?.data_path ?? ''}
                onchange={(v) => updateValidation('data_path', v || null)}
            />
        </FormField>

        <FormField label="Validation Interval Steps" description="Steps between validation runs">
            <NumberInput
                value={validation?.interval_steps ?? 500}
                min={1}
                onchange={(v) => updateValidation('interval_steps', v)}
            />
        </FormField>

        <FormField label="Validation Num Steps" description="Steps per validation run">
            <NumberInput
                value={validation?.num_steps ?? 10}
                min={1}
                onchange={(v) => updateValidation('num_steps', v)}
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
