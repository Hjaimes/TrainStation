<script lang="ts">
    import { configState } from '$lib/stores/config';
    import { FormField, SelectInput, PathInput, NumberInput, TextInput, ToggleInput } from '$lib/components/form';
    import type { DataConfig, DatasetEntry } from '$lib/types/config';

    let cfg = $derived($configState);
    let data = $derived(cfg.config?.data);
    let batchSize = $derived(cfg.config?.training?.batch_size);
    let showAdvanced = $state(false);

    function update(field: keyof DataConfig, value: unknown) {
        configState.updateSection('data', { [field]: value } as Partial<DataConfig>);
    }

    function updateDataset(index: number, field: keyof DatasetEntry, value: unknown) {
        const datasets = [...(cfg.config?.data.datasets ?? [])];
        datasets[index] = { ...datasets[index], [field]: value };
        configState.updateSection('data', { datasets });
    }

    function addDataset() {
        const datasets = [...(cfg.config?.data.datasets ?? [])];
        datasets.push({
            path: '',
            caption_extension: '.txt',
            repeats: 10,
            weight: 1.0,
            is_video: false,
            num_frames: 1,
            frame_extraction: 'uniform'
        });
        configState.updateSection('data', { datasets });
    }

    function removeDataset(index: number) {
        const datasets = (cfg.config?.data.datasets ?? []).filter((_: DatasetEntry, i: number) => i !== index);
        configState.updateSection('data', { datasets });
    }
</script>

<div class="section">
    <h2 class="section-title">Data</h2>

    <div class="fields">
        <FormField label="Dataset Config Path" description="Path to TOML dataset config (Musubi compat)">
            <PathInput
                value={data?.dataset_config_path ?? ''}
                mode="file"
                extensions={['toml']}
                onchange={(v) => update('dataset_config_path', v || null)}
            />
        </FormField>

        <FormField label="Batch Size" description="Training batch size">
            <NumberInput
                value={batchSize ?? 1}
                min={1}
                onchange={(v) => configState.updateSection('training', { batch_size: v })}
            />
        </FormField>
    </div>

    <div class="dataset-list">
        <h3 class="subsection-title">Datasets</h3>
        {#each (data?.datasets ?? []) as ds, i}
            <div class="dataset-row">
                <FormField label="Path">
                    <PathInput value={ds.path} onchange={(v) => updateDataset(i, 'path', v)} />
                </FormField>
                <FormField label="Repeats">
                    <NumberInput value={ds.repeats} min={1} onchange={(v) => updateDataset(i, 'repeats', v)} />
                </FormField>
                <FormField label="Weight">
                    <NumberInput value={ds.weight} min={0} step={0.1} onchange={(v) => updateDataset(i, 'weight', v)} />
                </FormField>
                <button class="btn-delete" type="button" onclick={() => removeDataset(i)}>&#x2715;</button>
            </div>
        {/each}
        <button type="button" class="btn-add" onclick={addDataset}>+ Add Dataset</button>
    </div>

    <button class="toggle-advanced" type="button" onclick={() => showAdvanced = !showAdvanced}>
        {showAdvanced ? '\u25BE Hide' : '\u25B8 Show'} Advanced Settings
    </button>

    {#if showAdvanced}
    <div class="fields advanced-fields">
        <FormField label="Resolution" description="Training resolution in pixels">
            <NumberInput
                value={data?.resolution ?? 512}
                min={64}
                step={64}
                onchange={(v) => update('resolution', v)}
            />
        </FormField>

        <FormField label="Enable Bucket" description="Use aspect ratio bucketing">
            <ToggleInput
                value={data?.enable_bucket ?? true}
                onchange={(v) => update('enable_bucket', v)}
            />
        </FormField>

        <FormField label="Num Workers" description="Data loader worker count">
            <NumberInput
                value={data?.num_workers ?? 4}
                min={0}
                max={16}
                onchange={(v) => update('num_workers', v)}
            />
        </FormField>

        <FormField label="Persistent Workers" description="Keep workers alive between epochs">
            <ToggleInput
                value={data?.persistent_workers ?? true}
                onchange={(v) => update('persistent_workers', v)}
            />
        </FormField>

        <FormField label="Flip Augmentation" description="Random horizontal flip">
            <ToggleInput
                value={data?.flip_aug ?? false}
                onchange={(v) => update('flip_aug', v)}
            />
        </FormField>

        <FormField label="Crop Jitter" description="Random crop jitter amount">
            <NumberInput
                value={data?.crop_jitter ?? 0}
                min={0}
                onchange={(v) => update('crop_jitter', v)}
            />
        </FormField>

        <FormField label="Shuffle Tags" description="Randomly shuffle caption tags">
            <ToggleInput
                value={data?.shuffle_tags ?? false}
                onchange={(v) => update('shuffle_tags', v)}
            />
        </FormField>

        <FormField label="Keep Tags Count" description="Number of leading tags to keep unshuffled">
            <NumberInput
                value={data?.keep_tags_count ?? 0}
                min={0}
                onchange={(v) => update('keep_tags_count', v)}
            />
        </FormField>

        <FormField label="Token Dropout Rate" description="Probability of dropping caption tokens">
            <NumberInput
                value={data?.token_dropout_rate ?? 0}
                min={0}
                max={1}
                step={0.01}
                onchange={(v) => update('token_dropout_rate', v)}
            />
        </FormField>

        <FormField label="Caption Delimiter" description="Delimiter between caption tags">
            <TextInput
                value={data?.caption_delimiter ?? ', '}
                onchange={(v) => update('caption_delimiter', v)}
            />
        </FormField>

        <FormField label="Masked Training" description="Enable masked loss training">
            <ToggleInput
                value={data?.masked_training ?? false}
                onchange={(v) => update('masked_training', v)}
            />
        </FormField>

        <FormField label="Mask Weight" description="Weight for masked regions">
            <NumberInput
                value={data?.mask_weight ?? 1.0}
                min={0}
                step={0.1}
                onchange={(v) => update('mask_weight', v)}
            />
        </FormField>

        <FormField label="Regularization Data Path" description="Path to regularization images">
            <PathInput
                value={data?.reg_data_path ?? ''}
                onchange={(v) => update('reg_data_path', v || null)}
            />
        </FormField>

        <FormField label="Prior Loss Weight" description="Weight for prior preservation loss">
            <NumberInput
                value={data?.prior_loss_weight ?? 1.0}
                min={0}
                step={0.1}
                onchange={(v) => update('prior_loss_weight', v)}
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
    .dataset-list {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .dataset-row {
        display: grid;
        grid-template-columns: 2fr 1fr 1fr auto;
        gap: 12px;
        align-items: end;
        padding: 12px;
        background: var(--bg-secondary);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius);
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
