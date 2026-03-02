<script lang="ts">
    import { onMount } from 'svelte';
    import SampleGrid from '$lib/components/samples/SampleGrid.svelte';
    import Lightbox from '$lib/components/samples/Lightbox.svelte';

    interface Sample {
        filename: string;
        absolute_path: string;
        size: number;
        modified: number;
    }

    let samples = $state<Sample[]>([]);
    let selectedSample = $state<Sample | null>(null);
    let loading = $state(true);
    let searchQuery = $state('');

    let filteredSamples = $derived(
        searchQuery
            ? samples.filter(s => s.filename.toLowerCase().includes(searchQuery.toLowerCase()))
            : samples
    );

    onMount(async () => {
        await refresh();
    });

    async function refresh() {
        loading = true;
        try {
            const resp = await fetch('/api/samples');
            samples = await resp.json();
        } catch {
            samples = [];
        }
        loading = false;
    }
</script>

<div class="samples-page">
    <div class="page-header">
        <h1>Samples</h1>
        <div class="header-actions">
            <input
                type="text"
                class="search-input"
                placeholder="Search samples..."
                bind:value={searchQuery}
            />
            <button type="button" onclick={refresh}>Refresh</button>
        </div>
    </div>

    {#if loading}
        <div class="loading">Loading samples...</div>
    {:else if samples.length === 0}
        <div class="empty-state">
            <span class="empty-icon">&#9714;</span>
            <p>No samples found. Generate samples during training to see them here.</p>
        </div>
    {:else}
        <SampleGrid samples={filteredSamples} onselect={(s) => selectedSample = s} />
    {/if}

    <Lightbox
        sample={selectedSample}
        samples={filteredSamples}
        onclose={() => selectedSample = null}
        onnavigate={(s) => selectedSample = s}
    />
</div>

<style>
    .samples-page { display: flex; flex-direction: column; height: 100%; gap: 20px; }
    .page-header { display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
    h1 { font-family: var(--font-mono); font-size: 18px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--text-primary); }
    .header-actions { display: flex; gap: 8px; align-items: center; }
    .search-input {
        font-family: var(--font-mono); font-size: 12px;
        background: var(--bg-primary); color: var(--text-primary);
        border: 1px solid var(--border); border-radius: var(--radius);
        padding: 6px 10px; width: 220px;
    }
    .search-input:focus { outline: none; border-color: var(--accent-dim); }
    .search-input::placeholder { color: var(--text-muted); }
    .loading { color: var(--text-muted); font-style: italic; padding: 40px 0; text-align: center; }
    .empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; flex: 1; gap: 12px; color: var(--text-muted); }
    .empty-icon { font-size: 48px; opacity: 0.2; }
    .empty-state p { font-size: 14px; font-style: italic; }
</style>
