<script lang="ts">
    import { configState } from '$lib/stores/config';
    import { FormField, SelectInput, PathInput, NumberInput, TextInput, ToggleInput } from '$lib/components/form';
    import type { NetworkConfig } from '$lib/types/config';

    let cfg = $derived($configState);
    let network = $derived(cfg.config?.network);
    let training = $derived(cfg.config?.training);
    let isFullFinetune = $derived(training?.method === 'full_finetune');
    let showAdvanced = $state(false);

    function update(field: keyof NetworkConfig, value: unknown) {
        configState.updateSection('network', { [field]: value } as Partial<NetworkConfig>);
    }

    /** Parse comma-separated string into string array */
    function parseList(input: string): string[] {
        if (!input.trim()) return [];
        return input.split(',').map(s => s.trim()).filter(Boolean);
    }

    /** Parse comma-separated string into number array or null */
    function parseNumberList(input: string): number[] | null {
        if (!input.trim()) return null;
        const nums = input.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
        return nums.length > 0 ? nums : null;
    }

    /** Format array to comma-separated display string */
    function formatList(arr: string[] | null | undefined): string {
        return arr?.join(', ') ?? '';
    }

    function formatNumberList(arr: number[] | null | undefined): string {
        return arr?.join(', ') ?? '';
    }
</script>

<div class="section">
    <h2 class="section-title">Network</h2>

    {#if isFullFinetune}
        <div class="info-message">
            <span class="info-icon">i</span>
            <span>Network settings do not apply when using full fine-tuning. Switch to LoRA, LoHa, or LoKr training to configure network parameters.</span>
        </div>
    {:else}
        <div class="fields">
            <FormField label="Network Type" description="LoRA variant to use for training">
                <SelectInput
                    value={network?.network_type ?? 'lora'}
                    options={[
                        { value: 'lora', label: 'LoRA' },
                        { value: 'loha', label: 'LoHa' },
                        { value: 'lokr', label: 'LoKr' },
                    ]}
                    onchange={(v) => update('network_type', v)}
                />
            </FormField>

            <FormField label="Rank" description="Network rank (dimensionality)">
                <NumberInput
                    value={network?.rank ?? 16}
                    min={1}
                    onchange={(v) => update('rank', v)}
                />
            </FormField>

            <FormField label="Alpha" description="Scaling factor (typically equal to rank)">
                <NumberInput
                    value={network?.alpha ?? 16}
                    step={0.1}
                    onchange={(v) => update('alpha', v)}
                />
            </FormField>

            <FormField label="Save Dtype" description="Dtype for saving network weights">
                <SelectInput
                    value={network?.save_dtype ?? 'none'}
                    options={[
                        { value: 'none', label: 'None (same as training)' },
                        { value: 'bf16', label: 'BF16' },
                        { value: 'fp16', label: 'FP16' },
                        { value: 'fp32', label: 'FP32' },
                    ]}
                    onchange={(v) => update('save_dtype', v === 'none' ? null : v)}
                />
            </FormField>
        </div>

        <button class="toggle-advanced" type="button" onclick={() => showAdvanced = !showAdvanced}>
            {showAdvanced ? '\u25BE Hide' : '\u25B8 Show'} Advanced Settings
        </button>

        {#if showAdvanced}
        <div class="fields advanced-fields">
            <FormField label="Use DoRA" description="Weight-Decomposed Low-Rank Adaptation">
                <ToggleInput
                    value={network?.use_dora ?? false}
                    onchange={(v) => update('use_dora', v)}
                />
            </FormField>

            <FormField label="Dropout" description="Network dropout rate (0 = disabled)">
                <NumberInput
                    value={network?.dropout ?? 0}
                    min={0}
                    max={1}
                    step={0.01}
                    onchange={(v) => update('dropout', v || null)}
                />
            </FormField>

            <FormField label="Rank Dropout" description="Per-rank dropout rate">
                <NumberInput
                    value={network?.rank_dropout ?? 0}
                    min={0}
                    max={1}
                    step={0.01}
                    onchange={(v) => update('rank_dropout', v || null)}
                />
            </FormField>

            <FormField label="Module Dropout" description="Per-module dropout rate">
                <NumberInput
                    value={network?.module_dropout ?? 0}
                    min={0}
                    max={1}
                    step={0.01}
                    onchange={(v) => update('module_dropout', v || null)}
                />
            </FormField>

            <FormField label="Scale Weight Norms" description="Max norm constraint (empty = disabled)">
                <NumberInput
                    value={network?.scale_weight_norms ?? 0}
                    min={0}
                    step={0.1}
                    onchange={(v) => update('scale_weight_norms', v || null)}
                />
            </FormField>

            <FormField label="LoRA+ LR Ratio" description="Learning rate ratio for LoRA+ (empty = disabled)">
                <NumberInput
                    value={network?.loraplus_lr_ratio ?? 0}
                    min={0}
                    step={0.1}
                    onchange={(v) => update('loraplus_lr_ratio', v || null)}
                />
            </FormField>

            <FormField label="Network Weights Path" description="Resume from existing network weights">
                <PathInput
                    value={network?.network_weights ?? ''}
                    mode="file"
                    extensions={['safetensors', 'ckpt', 'pt']}
                    onchange={(v) => update('network_weights', v || null)}
                />
            </FormField>

            <FormField label="Block LR Multipliers" description="Comma-separated per-block LR multipliers">
                <TextInput
                    value={formatNumberList(network?.block_lr_multipliers)}
                    placeholder="e.g. 1.0, 0.5, 0.5, 1.0"
                    onchange={(v) => update('block_lr_multipliers', parseNumberList(v))}
                />
            </FormField>

            <FormField label="Exclude Patterns" description="Comma-separated module patterns to exclude">
                <TextInput
                    value={formatList(network?.exclude_patterns)}
                    placeholder="e.g. time_embed, label_embed"
                    onchange={(v) => update('exclude_patterns', parseList(v))}
                />
            </FormField>

            <FormField label="Include Patterns" description="Comma-separated module patterns to include (empty = all)">
                <TextInput
                    value={formatList(network?.include_patterns)}
                    placeholder="e.g. attn, mlp"
                    onchange={(v) => update('include_patterns', parseList(v))}
                />
            </FormField>
        </div>
        {/if}
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
    .info-message {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 12px 16px;
        background: rgba(59, 158, 255, 0.06);
        border: 1px solid rgba(59, 158, 255, 0.15);
        border-radius: var(--radius);
        font-size: 13px;
        color: var(--text-secondary);
        line-height: 1.5;
    }
    .info-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: rgba(59, 158, 255, 0.15);
        color: var(--accent);
        font-family: var(--font-mono);
        font-size: 11px;
        font-weight: 700;
        flex-shrink: 0;
        margin-top: 1px;
    }
</style>
