# Phase 6: Full UI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Build a production-quality web UI with form-based configuration, live training dashboard, sample gallery, job queue, and preset system.

**Architecture:** SvelteKit 5 (Svelte 5 runes: `$state()`, `$derived()`, `$effect()`) frontend compiled to static SPA, served by FastAPI backend. WebSocket for live training events. Chart.js for loss plots. All config editing through typed form fields bound to a Pydantic-mirroring TypeScript config store.

**Tech Stack:** SvelteKit 2 + Svelte 5, FastAPI, Chart.js, WebSocket, YAML (presets)

**Starting State:** Working skeleton with: sidebar layout, `/training` page (metrics grid, raw JSON config editor, log viewer), WebSocket training event stream, `training.ts` and `config.ts` stores, 5 Svelte components, 3 backend route files. 1099 tests passing.

**Design Doc:** `docs/plans/2026-03-02-phase6-ui-design.md`

---

## Task Overview

| # | Task | Scope | Dependencies |
|---|------|-------|-------------|
| 1 | Navigation & routing overhaul | Layout + 5 page shells | None |
| 2 | Shared form components | 6 reusable input components | None |
| 3 | Config store overhaul | Typed TypeScript store | None |
| 4 | Configure page shell + section tabs | Page layout + tab switching | 1, 2, 3 |
| 5 | Model & Network form sections | Two config sections | 2, 4 |
| 6 | Data & Training form sections | Two config sections | 2, 4 |
| 7 | Sampling & Output form sections | Two config sections + Raw tab | 2, 4 |
| 8 | Monitor page — loss chart | Chart.js integration | 1, 3 |
| 9 | Monitor page — enhanced dashboard | Full 2x2 grid layout | 8 |
| 10 | Preset system — backend | API + file persistence | None |
| 11 | Preset system — frontend | Dropdown + save dialog | 4, 10 |
| 12 | Pre-flight validation modal | Validation UI before training | 3, 4 |
| 13 | Queue manager — backend | QueueManager + API routes | None |
| 14 | Queue page — frontend | Job list UI | 1, 13 |
| 15 | Samples API — backend | File listing + serving | None |
| 16 | Samples page — gallery + lightbox | Thumbnail grid + viewer | 1, 15 |
| 17 | Settings page | App-level preferences | 1 |
| 18 | Polish & integration testing | Wire everything, test E2E | All |

---

## Task 1: Navigation & Routing Overhaul

**Files:**
- Modify: `ui/frontend/src/routes/+layout.svelte`
- Create: `ui/frontend/src/routes/configure/+page.svelte`
- Create: `ui/frontend/src/routes/monitor/+page.svelte`
- Create: `ui/frontend/src/routes/samples/+page.svelte`
- Create: `ui/frontend/src/routes/queue/+page.svelte`
- Create: `ui/frontend/src/routes/settings/+page.svelte`
- Modify: `ui/frontend/src/routes/+page.svelte` (redirect to /configure)

**Step 1: Update sidebar navigation**

Replace the `navItems` array in `+layout.svelte` with the 5 new routes. Remove the disabled items and "Soon" badges:

```svelte
const navItems = [
    { label: 'Configure', href: '/configure', icon: '⚙' },
    { label: 'Monitor', href: '/monitor', icon: '◈' },
    { label: 'Samples', href: '/samples', icon: '◲' },
    { label: 'Queue', href: '/queue', icon: '☰' },
    { label: 'Settings', href: '/settings', icon: '⚡' },
];
```

Remove the `{#if item.disabled}` block and `.disabled` class since all items are now active.

**Step 2: Create placeholder page shells**

Each new page gets a minimal shell with the page title. Example for `/configure`:

```svelte
<!-- ui/frontend/src/routes/configure/+page.svelte -->
<div class="page">
    <div class="page-header">
        <h1>Configure</h1>
    </div>
    <div class="page-body">
        <p class="placeholder">Configuration form coming soon.</p>
    </div>
</div>

<style>
    .page { display: flex; flex-direction: column; height: 100%; gap: 20px; }
    .page-header { display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
    h1 { font-family: var(--font-mono); font-size: 18px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--text-primary); }
    .placeholder { color: var(--text-muted); font-style: italic; }
</style>
```

Repeat for monitor, samples, queue, settings (different titles).

**Step 3: Update root page to redirect**

```svelte
<!-- ui/frontend/src/routes/+page.svelte -->
<script lang="ts">
    import { goto } from '$app/navigation';
    import { onMount } from 'svelte';
    onMount(() => goto('/configure'));
</script>
```

**Step 4: Move existing training page to monitor**

Move the content from `ui/frontend/src/routes/training/+page.svelte` into `ui/frontend/src/routes/monitor/+page.svelte`. The old `/training` route can be deleted or redirect to `/monitor`.

**Step 5: Verify and commit**

Run: `cd ui/frontend && npm run build`
Expected: Build succeeds, all 5 routes accessible.

```bash
git add ui/frontend/src/
git commit -m "feat(ui): overhaul navigation with 5 main pages"
```

---

## Task 2: Shared Form Components

**Files:**
- Create: `ui/frontend/src/lib/components/form/FormField.svelte`
- Create: `ui/frontend/src/lib/components/form/TextInput.svelte`
- Create: `ui/frontend/src/lib/components/form/NumberInput.svelte`
- Create: `ui/frontend/src/lib/components/form/SelectInput.svelte`
- Create: `ui/frontend/src/lib/components/form/ToggleInput.svelte`
- Create: `ui/frontend/src/lib/components/form/PathInput.svelte`
- Create: `ui/frontend/src/lib/components/form/index.ts`

All form inputs follow the same pattern: label, input element, optional description, optional error. They use Svelte 5 runes (`$state`, `$derived`) and bind values via `bind:value` or callback props.

**Step 1: Create FormField wrapper**

This is the layout container — label on top, input below, optional help text:

```svelte
<!-- ui/frontend/src/lib/components/form/FormField.svelte -->
<script lang="ts">
    interface Props {
        label: string;
        description?: string;
        error?: string;
        children: any;
    }
    let { label, description = '', error = '', children }: Props = $props();
</script>

<div class="form-field" class:has-error={!!error}>
    <label class="field-label">{label}</label>
    {@render children()}
    {#if description && !error}
        <span class="field-desc">{description}</span>
    {/if}
    {#if error}
        <span class="field-error">{error}</span>
    {/if}
</div>

<style>
    .form-field { display: flex; flex-direction: column; gap: 6px; }
    .field-label { font-family: var(--font-mono); font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-secondary); }
    .field-desc { font-size: 11px; color: var(--text-muted); }
    .field-error { font-size: 11px; color: var(--error); }
</style>
```

**Step 2: Create TextInput, NumberInput, SelectInput, ToggleInput, PathInput**

Each follows the same pattern. Example `NumberInput`:

```svelte
<!-- ui/frontend/src/lib/components/form/NumberInput.svelte -->
<script lang="ts">
    interface Props {
        value: number;
        min?: number;
        max?: number;
        step?: number;
        placeholder?: string;
        onchange?: (value: number) => void;
    }
    let { value = $bindable(), min, max, step = 1, placeholder = '', onchange }: Props = $props();

    function handleInput(e: Event) {
        const v = parseFloat((e.target as HTMLInputElement).value);
        if (!isNaN(v)) { value = v; onchange?.(v); }
    }
</script>

<input type="number" class="num-input" {value} {min} {max} {step} {placeholder} oninput={handleInput} />

<style>
    .num-input { font-family: var(--font-mono); font-size: 13px; background: var(--bg-primary); color: var(--text-primary); border: 1px solid var(--border); border-radius: var(--radius); padding: 8px 10px; width: 100%; }
    .num-input:focus { outline: none; border-color: var(--accent-dim); }
</style>
```

`SelectInput` takes `options: {value: string, label: string}[]`. `ToggleInput` is a styled checkbox toggle. `PathInput` is a TextInput with a folder icon hint. `TextInput` is a basic string input.

**Step 3: Create barrel export**

```ts
// ui/frontend/src/lib/components/form/index.ts
export { default as FormField } from './FormField.svelte';
export { default as TextInput } from './TextInput.svelte';
export { default as NumberInput } from './NumberInput.svelte';
export { default as SelectInput } from './SelectInput.svelte';
export { default as ToggleInput } from './ToggleInput.svelte';
export { default as PathInput } from './PathInput.svelte';
```

**Step 4: Verify and commit**

Run: `cd ui/frontend && npm run build`

```bash
git add ui/frontend/src/lib/components/form/
git commit -m "feat(ui): add shared form input components"
```

---

## Task 3: Config Store Overhaul

**Files:**
- Modify: `ui/frontend/src/lib/stores/config.ts`
- Create: `ui/frontend/src/lib/types/config.ts`

The current config store holds `Record<string, unknown>`. Replace with typed TypeScript interfaces mirroring the Pydantic schema.

**Step 1: Create config type definitions**

```ts
// ui/frontend/src/lib/types/config.ts
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
```

**Step 2: Rewrite config store with typed state**

```ts
// ui/frontend/src/lib/stores/config.ts
import { writable } from 'svelte/store';
import type { TrainConfig } from '$lib/types/config';

export interface ConfigState {
    config: TrainConfig | null;
    dirty: boolean;
    valid: boolean | null;
    errors: string[];
    warnings: string[];
}

function createConfigStore() {
    const { subscribe, update, set } = writable<ConfigState>({
        config: null,
        dirty: false,
        valid: null,
        errors: [],
        warnings: []
    });

    return {
        subscribe, update, set,
        /** Update a nested config section. Marks store as dirty. */
        updateSection<K extends keyof TrainConfig>(section: K, values: Partial<TrainConfig[K]>) {
            update(s => {
                if (!s.config) return s;
                return {
                    ...s,
                    dirty: true,
                    config: {
                        ...s.config,
                        [section]: { ...s.config[section], ...values }
                    }
                };
            });
        }
    };
}

export const configState = createConfigStore();

export async function loadDefaults(arch: string): Promise<TrainConfig> {
    const resp = await fetch(`/api/config/defaults/${arch}`);
    const data: TrainConfig = await resp.json();
    configState.set({ config: data, dirty: false, valid: null, errors: [], warnings: [] });
    return data;
}

export async function validateConfig(config: TrainConfig) {
    const resp = await fetch('/api/config/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config })
    });
    const result = await resp.json();
    configState.update(s => ({
        ...s,
        valid: result.valid,
        errors: result.errors,
        warnings: result.warnings
    }));
    return result;
}
```

**Step 3: Verify and commit**

Run: `cd ui/frontend && npm run build`

```bash
git add ui/frontend/src/lib/types/ ui/frontend/src/lib/stores/config.ts
git commit -m "feat(ui): typed config store mirroring Pydantic schema"
```

---

## Task 4: Configure Page Shell + Section Tabs

**Files:**
- Modify: `ui/frontend/src/routes/configure/+page.svelte`
- Create: `ui/frontend/src/lib/components/configure/SectionTabs.svelte`
- Create: `ui/frontend/src/lib/components/configure/ConfigureToolbar.svelte`

**Step 1: Build section tab bar**

Inner sidebar/tab bar for switching between config sections:

```svelte
<!-- ui/frontend/src/lib/components/configure/SectionTabs.svelte -->
<script lang="ts">
    interface Props {
        active: string;
        onselect: (section: string) => void;
    }
    let { active, onselect }: Props = $props();

    const sections = [
        { id: 'model', label: 'Model', icon: '◆' },
        { id: 'network', label: 'Network', icon: '◇' },
        { id: 'data', label: 'Data', icon: '▤' },
        { id: 'training', label: 'Training', icon: '⚡' },
        { id: 'sampling', label: 'Sampling', icon: '◲' },
        { id: 'output', label: 'Output', icon: '↗' },
        { id: 'raw', label: 'Raw Config', icon: '{ }' },
    ];
</script>

<div class="section-tabs">
    {#each sections as sec}
        <button
            class="tab" class:active={active === sec.id}
            onclick={() => onselect(sec.id)}
        >
            <span class="tab-icon">{sec.icon}</span>
            <span class="tab-label">{sec.label}</span>
        </button>
    {/each}
</div>
```

Style: vertical list on the left side, highlighted active tab.

**Step 2: Build toolbar (preset dropdown + validate + start buttons)**

```svelte
<!-- ui/frontend/src/lib/components/configure/ConfigureToolbar.svelte -->
<script lang="ts">
    interface Props {
        onvalidate: () => void;
        onstart: () => void;
        onqueue: () => void;
        valid: boolean | null;
    }
    let { onvalidate, onstart, onqueue, valid }: Props = $props();
</script>

<div class="toolbar">
    <div class="toolbar-left">
        <!-- Preset dropdown added in Task 11 -->
        <slot name="preset" />
    </div>
    <div class="toolbar-right">
        <button onclick={onvalidate}>Validate</button>
        {#if valid === true}
            <span class="valid-badge">✓</span>
        {:else if valid === false}
            <span class="invalid-badge">✗</span>
        {/if}
        <button class="btn-primary" onclick={onstart}>▶ Start Training</button>
        <button onclick={onqueue}>+ Add to Queue</button>
    </div>
</div>
```

**Step 3: Assemble Configure page**

```svelte
<!-- ui/frontend/src/routes/configure/+page.svelte -->
<script lang="ts">
    import { onMount } from 'svelte';
    import { configState, loadDefaults, validateConfig } from '$lib/stores/config';
    import { startTraining } from '$lib/stores/training';
    import SectionTabs from '$lib/components/configure/SectionTabs.svelte';
    import ConfigureToolbar from '$lib/components/configure/ConfigureToolbar.svelte';

    let activeSection = $state('model');
    let state = $derived($configState);

    onMount(async () => {
        if (!state.config) await loadDefaults('wan');
    });

    async function handleValidate() {
        if (state.config) await validateConfig(state.config);
    }

    async function handleStart() {
        if (state.config) {
            await handleValidate();
            if (state.valid) await startTraining(state.config as any);
        }
    }

    function handleQueue() {
        // Task 14 wires this up
    }
</script>

<div class="configure-page">
    <ConfigureToolbar onvalidate={handleValidate} onstart={handleStart} onqueue={handleQueue} valid={state.valid} />

    <div class="configure-body">
        <SectionTabs active={activeSection} onselect={(s) => activeSection = s} />

        <div class="section-content">
            {#if activeSection === 'model'}
                <p class="placeholder">Model section (Task 5)</p>
            {:else if activeSection === 'network'}
                <p class="placeholder">Network section (Task 5)</p>
            {:else if activeSection === 'data'}
                <p class="placeholder">Data section (Task 6)</p>
            {:else if activeSection === 'training'}
                <p class="placeholder">Training section (Task 6)</p>
            {:else if activeSection === 'sampling'}
                <p class="placeholder">Sampling section (Task 7)</p>
            {:else if activeSection === 'output'}
                <p class="placeholder">Output section (Task 7)</p>
            {:else if activeSection === 'raw'}
                <p class="placeholder">Raw config (Task 7)</p>
            {/if}
        </div>
    </div>
</div>
```

Layout: toolbar at top, then horizontal split — narrow section tabs on left, form content on right.

**Step 4: Verify and commit**

Run: `cd ui/frontend && npm run build`

```bash
git add ui/frontend/src/routes/configure/ ui/frontend/src/lib/components/configure/
git commit -m "feat(ui): configure page shell with section tabs and toolbar"
```

---

## Task 5: Model & Network Form Sections

**Files:**
- Create: `ui/frontend/src/lib/components/configure/ModelSection.svelte`
- Create: `ui/frontend/src/lib/components/configure/NetworkSection.svelte`
- Modify: `ui/frontend/src/routes/configure/+page.svelte` (wire in sections)

**Step 1: Create ModelSection**

Essential fields: architecture (dropdown from `/api/models`), base_model_path, dtype, quantization.
Advanced fields (behind toggle): gradient_checkpointing, activation_offloading, weight_bouncing, block_swap_count, vae_path, vae_dtype, attn_mode, compile_model.

```svelte
<!-- ui/frontend/src/lib/components/configure/ModelSection.svelte -->
<script lang="ts">
    import { configState } from '$lib/stores/config';
    import { FormField, SelectInput, TextInput, NumberInput, ToggleInput, PathInput } from '$lib/components/form';
    import type { ModelConfig } from '$lib/types/config';

    let state = $derived($configState);
    let model: ModelConfig = $derived(state.config?.model ?? {} as ModelConfig);
    let showAdvanced = $state(false);

    let architectures = $state<string[]>([]);
    import { onMount } from 'svelte';
    onMount(async () => {
        const resp = await fetch('/api/models');
        const data = await resp.json();
        architectures = data.models;
    });

    function update(field: string, value: unknown) {
        configState.updateSection('model', { [field]: value });
    }
</script>

<div class="section">
    <h2 class="section-title">Model</h2>

    <div class="fields">
        <FormField label="Architecture">
            <SelectInput value={model.architecture}
                options={architectures.map(a => ({value: a, label: a}))}
                onchange={(v) => update('architecture', v)} />
        </FormField>

        <FormField label="Base Model Path" description="Path to transformer/DiT/UNet weights">
            <PathInput value={model.base_model_path}
                onchange={(v) => update('base_model_path', v)} />
        </FormField>

        <FormField label="Model Dtype">
            <SelectInput value={model.dtype}
                options={[{value:'bf16',label:'BF16'},{value:'fp16',label:'FP16'},{value:'fp32',label:'FP32'}]}
                onchange={(v) => update('dtype', v)} />
        </FormField>

        <FormField label="Quantization">
            <SelectInput value={model.quantization ?? 'none'}
                options={[{value:'none',label:'None'},{value:'fp8',label:'FP8'},{value:'fp8_scaled',label:'FP8 Scaled'},{value:'nf4',label:'NF4'},{value:'int8',label:'INT8'}]}
                onchange={(v) => update('quantization', v === 'none' ? null : v)} />
        </FormField>
    </div>

    <button class="toggle-advanced" onclick={() => showAdvanced = !showAdvanced}>
        {showAdvanced ? '▾ Hide' : '▸ Show'} Advanced
    </button>

    {#if showAdvanced}
    <div class="fields advanced">
        <FormField label="Gradient Checkpointing">
            <ToggleInput value={model.gradient_checkpointing}
                onchange={(v) => update('gradient_checkpointing', v)} />
        </FormField>
        <FormField label="Block Swap Count" description="0 = disabled">
            <NumberInput value={model.block_swap_count} min={0} max={100}
                onchange={(v) => update('block_swap_count', v)} />
        </FormField>
        <!-- activation_offloading, weight_bouncing, vae_path, vae_dtype, attn_mode, compile_model -->
    </div>
    {/if}
</div>
```

Use the same pattern (essential + advanced toggle) for all fields listed in the design doc.

**Step 2: Create NetworkSection**

Essential: network_type (dropdown), rank, alpha, save_dtype.
Advanced: use_dora, dropout, rank_dropout, module_dropout, block_lr_multipliers, exclude/include_patterns.

The Network section should only show when `training.method !== 'full_finetune'`.

```svelte
<!-- conditionally rendered in configure page -->
{#if state.config?.training.method !== 'full_finetune'}
    <NetworkSection />
{/if}
```

**Step 3: Wire into configure page**

Replace the model/network placeholders in `configure/+page.svelte` with `<ModelSection />` and `<NetworkSection />` imports.

**Step 4: Verify and commit**

Run: `cd ui/frontend && npm run build`

```bash
git add ui/frontend/src/lib/components/configure/ModelSection.svelte
git add ui/frontend/src/lib/components/configure/NetworkSection.svelte
git add ui/frontend/src/routes/configure/+page.svelte
git commit -m "feat(ui): model and network config form sections"
```

---

## Task 6: Data & Training Form Sections

**Files:**
- Create: `ui/frontend/src/lib/components/configure/DataSection.svelte`
- Create: `ui/frontend/src/lib/components/configure/TrainingSection.svelte`
- Modify: `ui/frontend/src/routes/configure/+page.svelte`

**Step 1: Create DataSection**

Essential fields: dataset_config_path (PathInput), datasets (add/remove rows with path + repeats + weight), batch_size (from training config — displayed here for UX).
Advanced: enable_bucket, num_workers, persistent_workers, flip_aug, crop_jitter, shuffle_tags, keep_tags_count, token_dropout_rate, masked_training, mask_weight, reg_data_path, prior_loss_weight.

For the datasets list, use an "Add Dataset" button that appends a `DatasetEntry` with defaults:

```svelte
function addDataset() {
    configState.updateSection('data', {
        datasets: [...(state.config?.data.datasets ?? []),
            { path: '', caption_extension: '.txt', repeats: 10, weight: 1.0,
              is_video: false, num_frames: 1, frame_extraction: 'uniform' }]
    });
}
```

Each dataset entry is a row with path input, repeats number input, weight number input, and a delete button.

**Step 2: Create TrainingSection**

Essential: optimizer_type (from OptimizerConfig), learning_rate, scheduler_type, epochs or max_steps toggle, gradient_accumulation_steps.
Advanced: warmup_steps, min_lr_ratio, noise_offset, timestep_sampling, loss_type, weighting_scheme, p2_gamma, zero_terminal_snr, stochastic_rounding, fused_backward, lr_scaling, seed, max_grad_norm, guidance_scale, ema_enabled, ema_decay, etc.

Note: This section pulls fields from both `TrainingConfig` and `OptimizerConfig`. The `update` function should use `configState.updateSection('training', ...)` or `configState.updateSection('optimizer', ...)` depending on the field.

**Step 3: Wire into configure page and verify**

```bash
git add ui/frontend/src/lib/components/configure/DataSection.svelte
git add ui/frontend/src/lib/components/configure/TrainingSection.svelte
git add ui/frontend/src/routes/configure/+page.svelte
git commit -m "feat(ui): data and training config form sections"
```

---

## Task 7: Sampling, Output Sections + Raw Config Tab

**Files:**
- Create: `ui/frontend/src/lib/components/configure/SamplingSection.svelte`
- Create: `ui/frontend/src/lib/components/configure/OutputSection.svelte`
- Create: `ui/frontend/src/lib/components/configure/RawConfigEditor.svelte`
- Modify: `ui/frontend/src/routes/configure/+page.svelte`

**Step 1: Create SamplingSection**

Essential: enabled (toggle), prompts (add/remove rows — each row has prompt text + optional seed), sample_every_n_steps, width, height, num_frames.
Advanced: num_inference_steps, guidance_scale, sample_at_first, prompts_file.

Prompts should be an editable list. Each prompt row: text input (wide) + seed number input (narrow) + delete button. "Add Prompt" button at bottom.

**Step 2: Create OutputSection**

Essential: output_dir (PathInput), output_name, save_every_n_steps, max_keep_ckpts.
Advanced: save_every_n_epochs, logging_dir, log_with, vram_profiling, validation settings (enabled, data_path, interval_steps, num_steps).

**Step 3: Create RawConfigEditor**

JSON/YAML textarea view of the full config. Two-way: editing the raw text updates the config store (on blur/validate), and form changes reflect in the raw view.

```svelte
<!-- ui/frontend/src/lib/components/configure/RawConfigEditor.svelte -->
<script lang="ts">
    import { configState } from '$lib/stores/config';

    let state = $derived($configState);
    let rawText = $state('');
    let parseError = $state('');

    $effect(() => {
        if (state.config) {
            rawText = JSON.stringify(state.config, null, 2);
        }
    });

    function handleBlur() {
        try {
            const parsed = JSON.parse(rawText);
            configState.update(s => ({ ...s, config: parsed, dirty: true }));
            parseError = '';
        } catch (e) {
            parseError = (e as Error).message;
        }
    }
</script>

<div class="raw-editor">
    <textarea bind:value={rawText} onblur={handleBlur} rows="30" spellcheck="false"></textarea>
    {#if parseError}
        <div class="parse-error">{parseError}</div>
    {/if}
</div>
```

**Step 4: Wire all into configure page, verify and commit**

```bash
git add ui/frontend/src/lib/components/configure/SamplingSection.svelte
git add ui/frontend/src/lib/components/configure/OutputSection.svelte
git add ui/frontend/src/lib/components/configure/RawConfigEditor.svelte
git add ui/frontend/src/routes/configure/+page.svelte
git commit -m "feat(ui): sampling, output sections and raw config editor"
```

---

## Task 8: Monitor Page — Loss Chart

**Files:**
- Modify: `ui/frontend/package.json` (add chart.js dependency)
- Create: `ui/frontend/src/lib/components/monitor/LossChart.svelte`
- Modify: `ui/frontend/src/lib/stores/training.ts` (add lossHistory array)

**Step 1: Install Chart.js**

```bash
cd ui/frontend && npm install chart.js
```

**Step 2: Extend training store with loss history**

Add to the `TrainingState` interface and store:

```ts
// In training.ts TrainingState:
lossHistory: { step: number; loss: number; avgLoss: number }[];

// In StepEvent handler:
next.lossHistory = [...s.lossHistory, {
    step: event.step as number,
    loss: event.loss as number,
    avgLoss: event.avg_loss as number
}];
// Cap at 10000 points to prevent memory bloat
if (next.lossHistory.length > 10000) {
    next.lossHistory = next.lossHistory.slice(-10000);
}
```

On `TrainingStartedEvent`, clear `lossHistory: []`.

**Step 3: Create LossChart component**

```svelte
<!-- ui/frontend/src/lib/components/monitor/LossChart.svelte -->
<script lang="ts">
    import { Chart, registerables } from 'chart.js';
    import { trainingState } from '$lib/stores/training';
    import { onMount, onDestroy } from 'svelte';

    Chart.register(...registerables);

    let canvas: HTMLCanvasElement;
    let chart: Chart | null = null;
    let history = $derived($trainingState.lossHistory);

    onMount(() => {
        chart = new Chart(canvas, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Loss', data: [], borderColor: 'rgba(59,158,255,0.3)', borderWidth: 1, pointRadius: 0, tension: 0.1 },
                    { label: 'Avg Loss', data: [], borderColor: '#3b9eff', borderWidth: 2, pointRadius: 0, tension: 0.3 },
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: { display: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#556677', font: { family: 'JetBrains Mono', size: 10 } } },
                    y: { display: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#556677', font: { family: 'JetBrains Mono', size: 10 } } }
                },
                plugins: { legend: { display: false } }
            }
        });
    });

    $effect(() => {
        if (chart && history) {
            chart.data.labels = history.map(h => h.step);
            chart.data.datasets[0].data = history.map(h => h.loss);
            chart.data.datasets[1].data = history.map(h => h.avgLoss);
            chart.update('none');
        }
    });

    onDestroy(() => chart?.destroy());
</script>

<div class="chart-container">
    <canvas bind:this={canvas}></canvas>
</div>

<style>
    .chart-container { position: relative; width: 100%; height: 100%; min-height: 200px; padding: 12px; }
</style>
```

**Step 4: Verify and commit**

```bash
cd ui/frontend && npm run build
git add ui/frontend/package.json ui/frontend/package-lock.json
git add ui/frontend/src/lib/components/monitor/LossChart.svelte
git add ui/frontend/src/lib/stores/training.ts
git commit -m "feat(ui): loss chart with Chart.js on monitor page"
```

---

## Task 9: Monitor Page — Enhanced Dashboard

**Files:**
- Modify: `ui/frontend/src/routes/monitor/+page.svelte`
- Create: `ui/frontend/src/lib/components/monitor/SamplePreview.svelte`
- Modify: `ui/frontend/src/lib/components/MetricsDisplay.svelte` (add VRAM, ETA)
- Modify: `ui/frontend/src/lib/stores/training.ts` (add VRAM, ETA, latest sample)

**Step 1: Extend training store**

Add to `TrainingState`:

```ts
vramPeakMb: number;
etaSeconds: number;
latestSample: { path: string; prompt: string; step: number } | null;
```

In `StepEvent` handler, compute ETA from step rate:
```ts
const elapsed = (Date.now() / 1000) - s._startTime;
const stepsPerSec = next.step / elapsed;
next.etaSeconds = stepsPerSec > 0 ? (next.totalSteps - next.step) / stepsPerSec : 0;
```

Add a `SampleEvent` case:
```ts
case 'SampleEvent':
    next.latestSample = {
        path: event.path as string,
        prompt: event.prompt as string,
        step: event.step as number
    };
    break;
```

**Step 2: Enhance MetricsDisplay**

Add VRAM peak and ETA to the metric grid. Add an ETA card that shows `"2m 30s"` formatted time.

**Step 3: Create SamplePreview**

Small panel showing the latest generated sample thumbnail. Clicking navigates to `/samples`.

```svelte
<!-- ui/frontend/src/lib/components/monitor/SamplePreview.svelte -->
<script lang="ts">
    import { trainingState } from '$lib/stores/training';
    let sample = $derived($trainingState.latestSample);
</script>

<div class="sample-preview">
    {#if sample}
        <img src="/api/samples/{sample.path}" alt="Sample at step {sample.step}" />
        <div class="sample-info">
            <span class="sample-step">Step {sample.step}</span>
            <span class="sample-prompt">{sample.prompt}</span>
        </div>
    {:else}
        <div class="no-sample">No samples yet</div>
    {/if}
</div>
```

**Step 4: Assemble 2x2 monitor dashboard**

Update `monitor/+page.svelte` with the 2x2 grid layout from the design:
- Top-left: LossChart
- Top-right: MetricsDisplay (with controls)
- Bottom-left: LogViewer
- Bottom-right: SamplePreview

```svelte
<div class="dashboard">
    <div class="panel chart-panel">
        <div class="panel-header">Loss</div>
        <LossChart />
    </div>
    <div class="panel metrics-panel">
        <div class="panel-header">Metrics</div>
        <MetricsDisplay />
        <TrainingControls />
    </div>
    <div class="panel log-panel">
        <div class="panel-header">Logs</div>
        <LogViewer />
    </div>
    <div class="panel sample-panel">
        <div class="panel-header">Latest Sample</div>
        <SamplePreview />
    </div>
</div>
```

Grid CSS: `grid-template-columns: 1fr 320px; grid-template-rows: 1fr 1fr;`

**Step 5: Verify and commit**

```bash
cd ui/frontend && npm run build
git add ui/frontend/src/
git commit -m "feat(ui): enhanced monitor dashboard with 2x2 grid layout"
```

---

## Task 10: Preset System — Backend

**Files:**
- Create: `ui/presets.py` (~120 lines)
- Create: `ui/routes/presets.py` (~60 lines)
- Modify: `ui/server.py` (include presets router)
- Create: `presets/builtin/` directory with 2-3 example presets
- Create: `tests/test_presets.py`

**Step 1: Write tests**

```python
# tests/test_presets.py
"""Tests for preset system."""
import json, tempfile, os
from pathlib import Path

def test_preset_manager_list_builtin(tmp_path):
    from ui.presets import PresetManager
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    (builtin / "wan-lora-24gb.yaml").write_text("model:\\n  architecture: wan\\n")
    mgr = PresetManager(builtin_dir=str(builtin), user_dir=str(tmp_path / "user"))
    presets = mgr.list_presets()
    assert len(presets) == 1
    assert presets[0]["name"] == "wan-lora-24gb"
    assert presets[0]["category"] == "builtin"

def test_preset_manager_save_user(tmp_path):
    from ui.presets import PresetManager
    mgr = PresetManager(builtin_dir=str(tmp_path / "builtin"), user_dir=str(tmp_path / "user"))
    mgr.save_user_preset("my-preset", {"model": {"architecture": "wan"}})
    presets = mgr.list_presets()
    assert any(p["name"] == "my-preset" for p in presets)

def test_preset_manager_load_merges_defaults(tmp_path):
    from ui.presets import PresetManager
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    import yaml
    (builtin / "test.yaml").write_text(yaml.dump({"model": {"architecture": "wan", "base_model_path": "/path"}}))
    mgr = PresetManager(builtin_dir=str(builtin), user_dir=str(tmp_path / "user"))
    config = mgr.load_preset("builtin", "test")
    # Should have defaults merged in
    assert config["model"]["architecture"] == "wan"
    assert "training" in config  # Default section present

def test_preset_manager_delete_user(tmp_path):
    from ui.presets import PresetManager
    mgr = PresetManager(builtin_dir=str(tmp_path / "builtin"), user_dir=str(tmp_path / "user"))
    mgr.save_user_preset("delete-me", {"model": {"architecture": "wan"}})
    mgr.delete_user_preset("delete-me")
    presets = mgr.list_presets()
    assert not any(p["name"] == "delete-me" for p in presets)
```

**Step 2: Implement PresetManager**

```python
# ui/presets.py
"""Preset system — builtin + user presets as partial YAML files."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
import yaml

from trainer.config.schema import TrainConfig, ModelConfig

logger = logging.getLogger(__name__)

class PresetManager:
    def __init__(self, builtin_dir: str = "presets/builtin", user_dir: str = "presets/user"):
        self._builtin = Path(builtin_dir)
        self._user = Path(user_dir)
        self._user.mkdir(parents=True, exist_ok=True)

    def list_presets(self) -> list[dict[str, str]]:
        result = []
        for d, category in [(self._builtin, "builtin"), (self._user, "user")]:
            if not d.exists(): continue
            for f in sorted(d.glob("*.yaml")):
                result.append({"name": f.stem, "category": category, "filename": f.name})
        return result

    def load_preset(self, category: str, name: str) -> dict[str, Any]:
        d = self._builtin if category == "builtin" else self._user
        path = d / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Preset not found: {category}/{name}")
        with open(path) as f:
            partial = yaml.safe_load(f) or {}
        return self._merge_with_defaults(partial)

    def save_user_preset(self, name: str, config: dict[str, Any]) -> None:
        path = self._user / f"{name}.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def delete_user_preset(self, name: str) -> None:
        path = self._user / f"{name}.yaml"
        if path.exists():
            path.unlink()

    def _merge_with_defaults(self, partial: dict[str, Any]) -> dict[str, Any]:
        arch = partial.get("model", {}).get("architecture", "wan")
        base_model_path = partial.get("model", {}).get("base_model_path", "<select model>")
        defaults = TrainConfig(
            model=ModelConfig(architecture=arch, base_model_path=base_model_path)
        ).model_dump()
        return _deep_merge(defaults, partial)

def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
```

**Step 3: Create API routes**

```python
# ui/routes/presets.py
from fastapi import APIRouter, HTTPException
router = APIRouter(prefix="/api/presets", tags=["presets"])

def _get_manager(request):
    # Lazy init on app state
    if not hasattr(request.app.state, 'preset_manager'):
        from ui.presets import PresetManager
        request.app.state.preset_manager = PresetManager()
    return request.app.state.preset_manager

@router.get("")
async def list_presets(request):
    return _get_manager(request).list_presets()

@router.get("/{category}/{name}")
async def load_preset(category: str, name: str, request):
    try:
        return _get_manager(request).load_preset(category, name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/user")
async def save_preset(body: dict, request):
    name = body.get("name", "").strip()
    config = body.get("config", {})
    if not name: raise HTTPException(400, "Name required")
    _get_manager(request).save_user_preset(name, config)
    return {"status": "saved"}

@router.delete("/user/{name}")
async def delete_preset(name: str, request):
    _get_manager(request).delete_user_preset(name)
    return {"status": "deleted"}
```

**Step 4: Include router in server.py, run tests, commit**

Add `from ui.routes.presets import router as presets_router` and `app.include_router(presets_router)` to `server.py`.

Run: `python -m pytest tests/test_presets.py -v`

```bash
git add ui/presets.py ui/routes/presets.py ui/server.py presets/ tests/test_presets.py
git commit -m "feat(ui): preset system backend with YAML file persistence"
```

---

## Task 11: Preset System — Frontend

**Files:**
- Create: `ui/frontend/src/lib/components/configure/PresetDropdown.svelte`
- Create: `ui/frontend/src/lib/components/configure/SavePresetDialog.svelte`
- Modify: `ui/frontend/src/lib/components/configure/ConfigureToolbar.svelte`
- Modify: `ui/frontend/src/routes/configure/+page.svelte`

**Step 1: Create PresetDropdown**

Fetches presets from `/api/presets`, groups by builtin/user, loads selected preset into config store:

```svelte
<script lang="ts">
    import { onMount } from 'svelte';
    import { configState } from '$lib/stores/config';

    interface Preset { name: string; category: string; }
    let presets = $state<Preset[]>([]);
    let selected = $state('');

    onMount(async () => {
        const resp = await fetch('/api/presets');
        presets = await resp.json();
    });

    async function loadPreset(category: string, name: string) {
        const resp = await fetch(`/api/presets/${category}/${name}`);
        const config = await resp.json();
        configState.set({ config, dirty: false, valid: null, errors: [], warnings: [] });
        selected = `${category}/${name}`;
    }
</script>

<select class="preset-select" value={selected} onchange={(e) => {
    const [cat, name] = (e.target as HTMLSelectElement).value.split('/');
    if (cat && name) loadPreset(cat, name);
}}>
    <option value="">Select preset...</option>
    <optgroup label="Built-in">
        {#each presets.filter(p => p.category === 'builtin') as p}
            <option value="builtin/{p.name}">{p.name}</option>
        {/each}
    </optgroup>
    <optgroup label="My Presets">
        {#each presets.filter(p => p.category === 'user') as p}
            <option value="user/{p.name}">{p.name}</option>
        {/each}
    </optgroup>
</select>
```

**Step 2: Create SavePresetDialog**

Simple modal with name input and Save button. Sends `POST /api/presets/user`:

```svelte
<script lang="ts">
    import { configState } from '$lib/stores/config';
    interface Props { open: boolean; onclose: () => void; }
    let { open, onclose }: Props = $props();
    let name = $state('');

    async function save() {
        const state = $configState;
        if (!name.trim() || !state.config) return;
        await fetch('/api/presets/user', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: name.trim(), config: state.config })
        });
        name = '';
        onclose();
    }
</script>

{#if open}
<div class="modal-overlay" onclick={onclose}>
    <div class="modal" onclick|stopPropagation>
        <h3>Save Preset</h3>
        <input type="text" bind:value={name} placeholder="Preset name..." />
        <div class="modal-actions">
            <button onclick={onclose}>Cancel</button>
            <button class="btn-primary" onclick={save}>Save</button>
        </div>
    </div>
</div>
{/if}
```

**Step 3: Wire into toolbar and configure page, verify and commit**

```bash
git add ui/frontend/src/lib/components/configure/PresetDropdown.svelte
git add ui/frontend/src/lib/components/configure/SavePresetDialog.svelte
git add ui/frontend/src/lib/components/configure/ConfigureToolbar.svelte
git add ui/frontend/src/routes/configure/+page.svelte
git commit -m "feat(ui): preset dropdown and save dialog on configure page"
```

---

## Task 12: Pre-flight Validation Modal

**Files:**
- Create: `ui/frontend/src/lib/components/configure/PreflightModal.svelte`
- Create: `ui/routes/preflight.py` (~50 lines)
- Modify: `ui/server.py` (include preflight router)
- Modify: `ui/frontend/src/routes/configure/+page.svelte`
- Create: `tests/test_preflight.py`

**Step 1: Create backend preflight endpoint**

```python
# ui/routes/preflight.py
"""Pre-flight validation — checks config, paths, cache files before training."""
from __future__ import annotations
import os
from fastapi import APIRouter
from pydantic import ValidationError

router = APIRouter(prefix="/api/preflight", tags=["preflight"])

@router.post("/check")
async def preflight_check(body: dict):
    from trainer.config.schema import TrainConfig
    from trainer.config.validation import validate_config
    from trainer.data.loader import check_cache_exists

    checks = []

    # 1. Config validation
    try:
        config = TrainConfig(**body.get("config", body))
        result = validate_config(config)
        if result.errors:
            checks.append({"name": "Config validation", "status": "error",
                           "message": "; ".join(i.message for i in result.errors)})
        elif result.warnings:
            checks.append({"name": "Config validation", "status": "warning",
                           "message": "; ".join(i.message for i in result.warnings)})
        else:
            checks.append({"name": "Config validation", "status": "ok", "message": "Valid"})
    except (ValidationError, ValueError) as e:
        checks.append({"name": "Config validation", "status": "error", "message": str(e)})
        return {"checks": checks, "can_start": False}

    # 2. Model path
    model_path = config.model.base_model_path
    if os.path.exists(model_path):
        checks.append({"name": "Model path", "status": "ok", "message": model_path})
    else:
        checks.append({"name": "Model path", "status": "error", "message": f"Not found: {model_path}"})

    # 3. Dataset / cache
    if config.data.dataset_config_path:
        if os.path.exists(config.data.dataset_config_path):
            checks.append({"name": "Dataset config", "status": "ok", "message": config.data.dataset_config_path})
        else:
            checks.append({"name": "Dataset config", "status": "error", "message": f"Not found: {config.data.dataset_config_path}"})
    elif config.data.datasets:
        for i, ds in enumerate(config.data.datasets):
            exists = os.path.isdir(ds.path)
            checks.append({"name": f"Dataset {i+1}", "status": "ok" if exists else "error",
                           "message": ds.path if exists else f"Not found: {ds.path}"})

    # 4. Output directory
    out_dir = config.saving.output_dir
    if os.path.isdir(out_dir) or os.access(os.path.dirname(out_dir) or ".", os.W_OK):
        checks.append({"name": "Output directory", "status": "ok", "message": out_dir})
    else:
        checks.append({"name": "Output directory", "status": "warning", "message": f"Will be created: {out_dir}"})

    can_start = all(c["status"] != "error" for c in checks)
    return {"checks": checks, "can_start": can_start}
```

**Step 2: Create PreflightModal component**

Shows the checklist with green/yellow/red indicators. "Start Training" button enabled only when `can_start` is true.

**Step 3: Wire into configure page**

The "Start Training" button in `ConfigureToolbar` now opens the preflight modal instead of starting directly.

**Step 4: Run tests, verify and commit**

```bash
python -m pytest tests/test_preflight.py -v
cd ui/frontend && npm run build
git add ui/routes/preflight.py ui/server.py tests/test_preflight.py
git add ui/frontend/src/lib/components/configure/PreflightModal.svelte
git add ui/frontend/src/routes/configure/+page.svelte
git commit -m "feat(ui): pre-flight validation modal with backend checks"
```

---

## Task 13: Queue Manager — Backend

**Files:**
- Create: `ui/queue.py` (~150 lines)
- Create: `ui/routes/queue.py` (~70 lines)
- Modify: `ui/server.py`
- Create: `tests/test_queue.py`

**Step 1: Write tests**

```python
# tests/test_queue.py
"""Tests for job queue manager."""
import json
from pathlib import Path

def test_queue_add_job(tmp_path):
    from ui.queue import QueueManager
    mgr = QueueManager(jobs_dir=str(tmp_path))
    job = mgr.add_job("test-run", {"model": {"architecture": "wan"}})
    assert job["status"] == "queued"
    assert job["name"] == "test-run"
    assert "id" in job

def test_queue_list_jobs(tmp_path):
    from ui.queue import QueueManager
    mgr = QueueManager(jobs_dir=str(tmp_path))
    mgr.add_job("job-1", {"model": {"architecture": "wan"}})
    mgr.add_job("job-2", {"model": {"architecture": "flux_2"}})
    jobs = mgr.list_jobs()
    assert len(jobs) == 2

def test_queue_remove_job(tmp_path):
    from ui.queue import QueueManager
    mgr = QueueManager(jobs_dir=str(tmp_path))
    job = mgr.add_job("delete-me", {})
    mgr.remove_job(job["id"])
    assert len(mgr.list_jobs()) == 0

def test_queue_reorder(tmp_path):
    from ui.queue import QueueManager
    mgr = QueueManager(jobs_dir=str(tmp_path))
    j1 = mgr.add_job("first", {})
    j2 = mgr.add_job("second", {})
    mgr.reorder_job(j2["id"], 0)
    jobs = mgr.list_jobs()
    assert jobs[0]["id"] == j2["id"]

def test_queue_clone_job(tmp_path):
    from ui.queue import QueueManager
    mgr = QueueManager(jobs_dir=str(tmp_path))
    j1 = mgr.add_job("original", {"model": {"architecture": "wan"}})
    j2 = mgr.clone_job(j1["id"])
    assert j2["name"] == "original (copy)"
    assert j2["config"] == j1["config"]
    assert j2["id"] != j1["id"]
```

**Step 2: Implement QueueManager**

```python
# ui/queue.py
"""Job queue manager with JSON file persistence."""
from __future__ import annotations
import json, logging, time, uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

class QueueManager:
    def __init__(self, jobs_dir: str = "jobs"):
        self._dir = Path(jobs_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._order_file = self._dir / "_order.json"

    def add_job(self, name: str, config: dict[str, Any]) -> dict:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id, "name": name, "status": "queued",
            "config": config, "created_at": time.time(),
            "started_at": None, "completed_at": None, "result": None,
        }
        self._save_job(job)
        order = self._load_order()
        order.append(job_id)
        self._save_order(order)
        return job

    def list_jobs(self) -> list[dict]:
        order = self._load_order()
        jobs = []
        for jid in order:
            job = self._load_job(jid)
            if job: jobs.append(job)
        # Also add completed jobs not in order
        for f in self._dir.glob("*.json"):
            if f.name.startswith("_"): continue
            jid = f.stem
            if jid not in order:
                job = self._load_job(jid)
                if job: jobs.append(job)
        return jobs

    def remove_job(self, job_id: str) -> None:
        path = self._dir / f"{job_id}.json"
        if path.exists(): path.unlink()
        order = self._load_order()
        self._save_order([j for j in order if j != job_id])

    def reorder_job(self, job_id: str, new_index: int) -> None:
        order = self._load_order()
        if job_id in order: order.remove(job_id)
        order.insert(new_index, job_id)
        self._save_order(order)

    def clone_job(self, job_id: str) -> dict:
        original = self._load_job(job_id)
        if not original: raise FileNotFoundError(f"Job {job_id} not found")
        return self.add_job(f"{original['name']} (copy)", original["config"])

    def get_next_queued(self) -> dict | None:
        for jid in self._load_order():
            job = self._load_job(jid)
            if job and job["status"] == "queued": return job
        return None

    def update_job(self, job_id: str, **fields) -> None:
        job = self._load_job(job_id)
        if job:
            job.update(fields)
            self._save_job(job)

    def _save_job(self, job: dict) -> None:
        path = self._dir / f"{job['id']}.json"
        path.write_text(json.dumps(job, indent=2))

    def _load_job(self, job_id: str) -> dict | None:
        path = self._dir / f"{job_id}.json"
        if not path.exists(): return None
        return json.loads(path.read_text())

    def _load_order(self) -> list[str]:
        if not self._order_file.exists(): return []
        return json.loads(self._order_file.read_text())

    def _save_order(self, order: list[str]) -> None:
        self._order_file.write_text(json.dumps(order))
```

**Step 3: Create API routes and wire into server**

```python
# ui/routes/queue.py
from fastapi import APIRouter, HTTPException, Request
router = APIRouter(prefix="/api/queue", tags=["queue"])

def _mgr(request: Request):
    if not hasattr(request.app.state, 'queue_manager'):
        from ui.queue import QueueManager
        request.app.state.queue_manager = QueueManager()
    return request.app.state.queue_manager

@router.get("")
async def list_jobs(request: Request):
    return _mgr(request).list_jobs()

@router.post("/add")
async def add_job(body: dict, request: Request):
    name = body.get("name", "Untitled")
    config = body.get("config", {})
    return _mgr(request).add_job(name, config)

@router.delete("/{job_id}")
async def remove_job(job_id: str, request: Request):
    _mgr(request).remove_job(job_id)
    return {"status": "removed"}

@router.post("/{job_id}/reorder")
async def reorder_job(job_id: str, body: dict, request: Request):
    _mgr(request).reorder_job(job_id, body.get("index", 0))
    return {"status": "reordered"}

@router.post("/{job_id}/clone")
async def clone_job(job_id: str, request: Request):
    return _mgr(request).clone_job(job_id)
```

**Step 4: Run tests, commit**

```bash
python -m pytest tests/test_queue.py -v
git add ui/queue.py ui/routes/queue.py ui/server.py tests/test_queue.py
git commit -m "feat(ui): job queue manager with JSON persistence and API"
```

---

## Task 14: Queue Page — Frontend

**Files:**
- Modify: `ui/frontend/src/routes/queue/+page.svelte`
- Create: `ui/frontend/src/lib/stores/queue.ts`

**Step 1: Create queue store**

```ts
// ui/frontend/src/lib/stores/queue.ts
import { writable } from 'svelte/store';

export interface Job {
    id: string; name: string; status: string;
    config: Record<string, unknown>;
    created_at: number; started_at: number | null;
    completed_at: number | null; result: any;
}

function createQueueStore() {
    const { subscribe, set } = writable<Job[]>([]);
    return {
        subscribe,
        async refresh() {
            const resp = await fetch('/api/queue');
            const jobs: Job[] = await resp.json();
            set(jobs);
        },
        async addJob(name: string, config: Record<string, unknown>) {
            await fetch('/api/queue/add', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, config })
            });
            this.refresh();
        },
        async removeJob(id: string) {
            await fetch(`/api/queue/${id}`, { method: 'DELETE' });
            this.refresh();
        },
        async cloneJob(id: string) {
            await fetch(`/api/queue/${id}/clone`, { method: 'POST' });
            this.refresh();
        }
    };
}

export const queueStore = createQueueStore();
```

**Step 2: Build queue page UI**

Three sections: Queued, Running, Completed. Each job shows name, architecture, status. Queued jobs have delete buttons. Completed jobs have "Clone" and "Re-run" buttons.

**Step 3: Verify and commit**

```bash
cd ui/frontend && npm run build
git add ui/frontend/src/lib/stores/queue.ts ui/frontend/src/routes/queue/+page.svelte
git commit -m "feat(ui): queue page with job list and management"
```

---

## Task 15: Samples API — Backend

**Files:**
- Create: `ui/routes/samples.py` (~40 lines)
- Modify: `ui/server.py`

**Step 1: Create samples endpoint**

Scans the output directory for sample images (*.png, *.jpg, *.mp4). Returns metadata list sorted most-recent-first.

```python
# ui/routes/samples.py
"""Sample gallery API — lists and serves generated sample files."""
from __future__ import annotations
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/samples", tags=["samples"])

SAMPLE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.mp4', '.gif'}

@router.get("")
async def list_samples(output_dir: str = "./output"):
    samples = []
    base = Path(output_dir)
    if not base.exists():
        return []
    for f in base.rglob("*"):
        if f.suffix.lower() in SAMPLE_EXTENSIONS and "sample" in f.name.lower():
            stat = f.stat()
            samples.append({
                "filename": str(f.relative_to(base)),
                "path": str(f),
                "size": stat.st_size,
                "modified": stat.st_mtime,
            })
    samples.sort(key=lambda s: s["modified"], reverse=True)
    return samples

@router.get("/file")
async def serve_sample(path: str):
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, "Sample not found")
    return FileResponse(str(p))
```

**Step 2: Wire into server, commit**

```bash
git add ui/routes/samples.py ui/server.py
git commit -m "feat(ui): samples API for listing and serving generated images"
```

---

## Task 16: Samples Page — Gallery + Lightbox

**Files:**
- Modify: `ui/frontend/src/routes/samples/+page.svelte`
- Create: `ui/frontend/src/lib/components/samples/SampleGrid.svelte`
- Create: `ui/frontend/src/lib/components/samples/Lightbox.svelte`

**Step 1: Create SampleGrid**

Fetches samples from API, displays as responsive thumbnail grid. Each card: image thumbnail, truncated prompt, step number, timestamp.

**Step 2: Create Lightbox**

Modal overlay with full-size image, full prompt, seed, step, dimensions. Left/right navigation with arrow keys.

**Step 3: Wire into samples page, verify and commit**

```bash
cd ui/frontend && npm run build
git add ui/frontend/src/routes/samples/ ui/frontend/src/lib/components/samples/
git commit -m "feat(ui): sample gallery with thumbnail grid and lightbox viewer"
```

---

## Task 17: Settings Page

**Files:**
- Modify: `ui/frontend/src/routes/settings/+page.svelte`

Minimal settings page with app-level preferences:
- Default architecture (dropdown)
- Default output directory
- Theme (just dark for V1, placeholder for light)
- WebSocket reconnect interval
- Max log entries

Store settings in `localStorage`. Simple form layout using the shared form components.

**Verify and commit:**

```bash
cd ui/frontend && npm run build
git add ui/frontend/src/routes/settings/+page.svelte
git commit -m "feat(ui): settings page with app preferences"
```

---

## Task 18: Polish & Integration Testing

**Files:**
- Modify: `tests/test_imports.py` (add canary imports for new UI modules)
- Modify: Various files for bug fixes found during integration

**Step 1: Add canary imports**

```python
# In tests/test_imports.py:
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
```

**Step 2: Run full test suite**

```bash
python -m pytest tests/ -v
```

All existing 1099+ tests must still pass, plus new test_presets and test_queue tests.

**Step 3: Build frontend**

```bash
cd ui/frontend && npm run build
```

**Step 4: Manual integration check**

Start the server and verify all pages render, navigation works, form fields bind correctly:

```bash
cd ui/frontend && npm run dev
# In another terminal:
uvicorn ui.server:app --port 8675
```

**Step 5: Fix any issues found, final commit**

```bash
git add -A
git commit -m "feat(ui): Phase 6 complete — full UI with configure, monitor, samples, queue, settings"
```

---

## Dependency Graph

```
Task 1 (Nav/Routing) ─────────┬──── Task 4 (Configure shell) ─┬── Task 5 (Model/Network)
                               │                               ├── Task 6 (Data/Training)
Task 2 (Form components) ─────┤                               └── Task 7 (Sampling/Output/Raw)
                               │
Task 3 (Config store) ────────┤    Task 8 (Loss chart) ── Task 9 (Monitor dashboard)
                               │
                               ├── Task 12 (Preflight modal)
                               │
Task 10 (Presets backend) ──── Task 11 (Presets frontend)
                               │
Task 13 (Queue backend) ────── Task 14 (Queue frontend)
                               │
Task 15 (Samples backend) ──── Task 16 (Samples gallery)
                               │
                               └── Task 17 (Settings)

Task 18 (Polish) depends on all above
```

**Parallel groups:**
- Tasks 1, 2, 3, 10, 13, 15 are fully independent (can run in parallel)
- Tasks 5, 6, 7 are independent of each other (all depend on 4)
- Tasks 8, 11, 14, 16, 17 are independent of each other (depend on their backends)
