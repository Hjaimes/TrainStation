<script lang="ts">
    import { onMount } from 'svelte';
    import { configState } from '$lib/stores/config';
    import type { TrainConfig } from '$lib/types/config';

    interface Preset {
        name: string;
        category: string;
        filename: string;
    }

    let presets = $state<Preset[]>([]);
    let selected = $state('');
    let loading = $state(false);

    onMount(async () => {
        await refresh();
    });

    export async function refresh() {
        try {
            const resp = await fetch('/api/presets');
            presets = await resp.json();
        } catch {
            presets = [];
        }
    }

    async function loadPreset(category: string, name: string) {
        loading = true;
        try {
            const resp = await fetch(`/api/presets/${category}/${name}`);
            const config: TrainConfig = await resp.json();
            configState.set({ config, dirty: false, valid: null, errors: [], warnings: [] });
            selected = `${category}/${name}`;
        } catch (e) {
            console.error('Failed to load preset:', e);
        }
        loading = false;
    }

    function handleChange(e: Event) {
        const val = (e.target as HTMLSelectElement).value;
        if (!val) return;
        const [cat, name] = val.split('/');
        if (cat && name) loadPreset(cat, name);
    }

    const builtins = $derived(presets.filter(p => p.category === 'builtin'));
    const userPresets = $derived(presets.filter(p => p.category === 'user'));
</script>

<div class="preset-dropdown">
    <select class="preset-select" value={selected} onchange={handleChange} disabled={loading}>
        <option value="">Load preset...</option>
        {#if builtins.length > 0}
            <optgroup label="Built-in">
                {#each builtins as p}
                    <option value="builtin/{p.name}">{p.name}</option>
                {/each}
            </optgroup>
        {/if}
        {#if userPresets.length > 0}
            <optgroup label="My Presets">
                {#each userPresets as p}
                    <option value="user/{p.name}">{p.name}</option>
                {/each}
            </optgroup>
        {/if}
    </select>
</div>

<style>
    .preset-dropdown { display: flex; align-items: center; }
    .preset-select {
        font-family: var(--font-mono);
        font-size: 12px;
        background: var(--bg-primary);
        color: var(--text-primary);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 6px 10px;
        min-width: 200px;
        cursor: pointer;
    }
    .preset-select:focus { outline: none; border-color: var(--accent-dim); }
    .preset-select option { background: var(--bg-secondary); color: var(--text-primary); }
    .preset-select:disabled { opacity: 0.5; }
</style>
