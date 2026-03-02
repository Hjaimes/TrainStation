<script lang="ts">
    import { onMount } from 'svelte';
    import { configState, loadDefaults } from '$lib/stores/config';
    import { FormField, SelectInput, PathInput, NumberInput, ToggleInput } from '$lib/components/form';
    import type { ModelConfig } from '$lib/types/config';

    let state = $derived($configState);
    let model = $derived(state.config?.model);
    let showAdvanced = $state(false);
    let architectures = $state<string[]>([]);

    onMount(async () => {
        try {
            const resp = await fetch('/api/models');
            const data = await resp.json();
            architectures = data.models;
        } catch {
            // API not available — fall back to current architecture
            if (model?.architecture) {
                architectures = [model.architecture];
            }
        }
    });

    function update(field: keyof ModelConfig, value: unknown) {
        configState.updateSection('model', { [field]: value } as Partial<ModelConfig>);
    }

    async function handleArchChange(arch: string) {
        await loadDefaults(arch);
    }
</script>

<div class="section">
    <h2 class="section-title">Model</h2>

    <div class="fields">
        <FormField label="Architecture" description="Model architecture to train">
            <SelectInput
                value={model?.architecture ?? 'wan'}
                options={architectures.map(a => ({ value: a, label: a }))}
                onchange={handleArchChange}
            />
        </FormField>

        <FormField label="Base Model Path" description="Path to transformer/DiT/UNet weights">
            <PathInput
                value={model?.base_model_path ?? ''}
                onchange={(v) => update('base_model_path', v)}
            />
        </FormField>

        <FormField label="Model Dtype">
            <SelectInput
                value={model?.dtype ?? 'bf16'}
                options={[
                    { value: 'bf16', label: 'BF16' },
                    { value: 'fp16', label: 'FP16' },
                    { value: 'fp32', label: 'FP32' },
                ]}
                onchange={(v) => update('dtype', v)}
            />
        </FormField>

        <FormField label="Quantization" description="Weight quantization for reduced VRAM">
            <SelectInput
                value={model?.quantization ?? 'none'}
                options={[
                    { value: 'none', label: 'None' },
                    { value: 'fp8', label: 'FP8' },
                    { value: 'fp8_scaled', label: 'FP8 Scaled' },
                    { value: 'nf4', label: 'NF4 (4-bit)' },
                    { value: 'int8', label: 'INT8' },
                ]}
                onchange={(v) => update('quantization', v === 'none' ? null : v)}
            />
        </FormField>
    </div>

    <button class="toggle-advanced" type="button" onclick={() => showAdvanced = !showAdvanced}>
        {showAdvanced ? '\u25BE Hide' : '\u25B8 Show'} Advanced Settings
    </button>

    {#if showAdvanced}
    <div class="fields advanced-fields">
        <FormField label="Gradient Checkpointing" description="Trade compute for VRAM savings">
            <ToggleInput
                value={model?.gradient_checkpointing ?? true}
                onchange={(v) => update('gradient_checkpointing', v)}
            />
        </FormField>

        <FormField label="Block Swap Count" description="Number of blocks to swap to CPU (0 = disabled)">
            <NumberInput
                value={model?.block_swap_count ?? 0}
                min={0}
                max={100}
                onchange={(v) => update('block_swap_count', v)}
            />
        </FormField>

        <FormField label="VAE Path" description="Custom VAE weights (leave empty for default)">
            <PathInput
                value={model?.vae_path ?? ''}
                onchange={(v) => update('vae_path', v || null)}
            />
        </FormField>

        <FormField label="VAE Dtype">
            <SelectInput
                value={model?.vae_dtype ?? 'bf16'}
                options={[
                    { value: 'bf16', label: 'BF16' },
                    { value: 'fp16', label: 'FP16' },
                ]}
                onchange={(v) => update('vae_dtype', v)}
            />
        </FormField>

        <FormField label="Attention Mode">
            <SelectInput
                value={model?.attn_mode ?? 'sdpa'}
                options={[
                    { value: 'sdpa', label: 'SDPA' },
                    { value: 'flash', label: 'Flash Attention' },
                    { value: 'xformers', label: 'xFormers' },
                ]}
                onchange={(v) => update('attn_mode', v)}
            />
        </FormField>

        <FormField label="Compile Model" description="torch.compile for potential speedup">
            <ToggleInput
                value={model?.compile_model ?? false}
                onchange={(v) => update('compile_model', v)}
            />
        </FormField>

        <FormField label="Activation Offloading" description="Offload activations to CPU during backward">
            <ToggleInput
                value={model?.activation_offloading ?? false}
                onchange={(v) => update('activation_offloading', v)}
            />
        </FormField>

        <FormField label="Weight Bouncing" description="Bounce weights between CPU and GPU per layer">
            <ToggleInput
                value={model?.weight_bouncing ?? false}
                onchange={(v) => update('weight_bouncing', v)}
            />
        </FormField>

        <FormField label="Split Attention">
            <ToggleInput
                value={model?.split_attn ?? false}
                onchange={(v) => update('split_attn', v)}
            />
        </FormField>
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
