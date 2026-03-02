<script lang="ts">
    import { configState } from '$lib/stores/config';
    import { FormField, PathInput, NumberInput, TextInput, ToggleInput } from '$lib/components/form';
    import type { SamplingConfig } from '$lib/types/config';

    let state = $derived($configState);
    let sampling = $derived(state.config?.sampling);
    let showAdvanced = $state(false);

    function update(field: keyof SamplingConfig, value: unknown) {
        configState.updateSection('sampling', { [field]: value } as Partial<SamplingConfig>);
    }

    function updatePrompt(index: number, value: string) {
        const prompts = [...(sampling?.prompts ?? [])];
        prompts[index] = value;
        configState.updateSection('sampling', { prompts });
    }

    function addPrompt() {
        const prompts = [...(sampling?.prompts ?? []), ''];
        configState.updateSection('sampling', { prompts });
    }

    function removePrompt(index: number) {
        const prompts = (sampling?.prompts ?? []).filter((_: string, i: number) => i !== index);
        configState.updateSection('sampling', { prompts });
    }
</script>

<div class="section">
    <h2 class="section-title">Sampling</h2>

    <div class="fields">
        <FormField label="Enabled" description="Generate sample images during training">
            <ToggleInput
                value={sampling?.enabled ?? false}
                onchange={(v) => update('enabled', v)}
            />
        </FormField>

        <FormField label="Sample Every N Steps" description="Steps between sample generation">
            <NumberInput
                value={sampling?.sample_every_n_steps ?? 100}
                min={1}
                onchange={(v) => update('sample_every_n_steps', v)}
            />
        </FormField>

        <FormField label="Width" description="Sample image width">
            <NumberInput
                value={sampling?.width ?? 512}
                min={64}
                step={64}
                onchange={(v) => update('width', v)}
            />
        </FormField>

        <FormField label="Height" description="Sample image height">
            <NumberInput
                value={sampling?.height ?? 512}
                min={64}
                step={64}
                onchange={(v) => update('height', v)}
            />
        </FormField>

        <FormField label="Num Frames" description="Number of video frames to sample">
            <NumberInput
                value={sampling?.num_frames ?? 1}
                min={1}
                onchange={(v) => update('num_frames', v)}
            />
        </FormField>
    </div>

    <div class="prompt-list">
        <h3 class="subsection-title">Prompts</h3>
        {#each (sampling?.prompts ?? []) as prompt, i}
            <div class="prompt-row">
                <div class="prompt-text">
                    <FormField label="Prompt {i + 1}">
                        <TextInput
                            value={prompt}
                            placeholder="Enter a prompt..."
                            onchange={(v) => updatePrompt(i, v)}
                        />
                    </FormField>
                </div>
                <button class="btn-delete" type="button" onclick={() => removePrompt(i)}>&#x2715;</button>
            </div>
        {/each}
        <button type="button" class="btn-add" onclick={addPrompt}>+ Add Prompt</button>
    </div>

    <button class="toggle-advanced" type="button" onclick={() => showAdvanced = !showAdvanced}>
        {showAdvanced ? '\u25BE Hide' : '\u25B8 Show'} Advanced Settings
    </button>

    {#if showAdvanced}
    <div class="fields advanced-fields">
        <FormField label="Num Inference Steps" description="Denoising steps for sampling">
            <NumberInput
                value={sampling?.num_inference_steps ?? 20}
                min={1}
                onchange={(v) => update('num_inference_steps', v)}
            />
        </FormField>

        <FormField label="Guidance Scale" description="CFG scale for sampling">
            <NumberInput
                value={sampling?.guidance_scale ?? 7.5}
                min={0}
                step={0.1}
                onchange={(v) => update('guidance_scale', v)}
            />
        </FormField>

        <FormField label="Sample At First" description="Generate a sample before training starts">
            <ToggleInput
                value={sampling?.sample_at_first ?? false}
                onchange={(v) => update('sample_at_first', v)}
            />
        </FormField>

        <FormField label="Prompts File" description="Path to file with prompts (one per line)">
            <PathInput
                value={sampling?.prompts_file ?? ''}
                onchange={(v) => update('prompts_file', v || null)}
            />
        </FormField>

        <FormField label="Sample Every N Epochs" description="Epochs between sample generation">
            <NumberInput
                value={sampling?.sample_every_n_epochs ?? 0}
                min={0}
                onchange={(v) => update('sample_every_n_epochs', v || null)}
            />
        </FormField>

        <FormField label="Seed" description="Seed for reproducible samples (empty for random)">
            <NumberInput
                value={sampling?.seed ?? 0}
                onchange={(v) => update('seed', v || null)}
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
    .subsection-title {
        font-family: var(--font-mono);
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-secondary);
        margin: 0 0 8px 0;
    }
    .fields {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
    }
    .prompt-list {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .prompt-row {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 12px;
        align-items: end;
        padding: 12px;
        background: var(--bg-secondary);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius);
    }
    .prompt-text {
        min-width: 0;
    }
    .btn-delete {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        background: none !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius);
        color: var(--text-muted);
        font-size: 14px;
        cursor: pointer;
        margin-bottom: 2px;
    }
    .btn-delete:hover {
        color: var(--error);
        border-color: var(--error) !important;
        background: rgba(239, 68, 68, 0.08) !important;
    }
    .btn-add {
        align-self: flex-start;
        background: none !important;
        border: 1px dashed var(--border-subtle) !important;
        border-radius: var(--radius);
        padding: 8px 16px !important;
        color: var(--text-muted);
        font-family: var(--font-mono);
        font-size: 12px;
        cursor: pointer;
    }
    .btn-add:hover {
        color: var(--accent);
        border-color: var(--accent) !important;
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
