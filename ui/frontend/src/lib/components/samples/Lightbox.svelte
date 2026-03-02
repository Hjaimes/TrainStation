<script lang="ts">
    interface Sample {
        filename: string;
        absolute_path: string;
        size: number;
        modified: number;
    }

    interface Props {
        sample: Sample | null;
        samples: Sample[];
        onclose: () => void;
        onnavigate: (sample: Sample) => void;
    }
    let { sample, samples, onclose, onnavigate }: Props = $props();

    let currentIndex = $derived(sample ? samples.findIndex(s => s.absolute_path === sample.absolute_path) : -1);
    let hasPrev = $derived(currentIndex > 0);
    let hasNext = $derived(currentIndex >= 0 && currentIndex < samples.length - 1);

    function prev() {
        if (hasPrev) onnavigate(samples[currentIndex - 1]);
    }
    function next() {
        if (hasNext) onnavigate(samples[currentIndex + 1]);
    }

    function handleKeydown(e: KeyboardEvent) {
        if (e.key === 'Escape') onclose();
        if (e.key === 'ArrowLeft') prev();
        if (e.key === 'ArrowRight') next();
    }
</script>

<svelte:window onkeydown={handleKeydown} />

{#if sample}
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div class="lightbox-overlay" onclick={onclose}>
        <!-- svelte-ignore a11y_click_events_have_key_events -->
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <div class="lightbox" onclick={(e) => e.stopPropagation()}>
            <div class="lightbox-header">
                <span class="lightbox-filename">{sample.filename}</span>
                <button type="button" class="lightbox-close" onclick={onclose}>✕</button>
            </div>

            <div class="lightbox-image">
                {#if hasPrev}
                    <button type="button" class="nav-btn nav-prev" onclick={prev}>&#8249;</button>
                {/if}
                <img
                    src="/api/samples/file?path={encodeURIComponent(sample.absolute_path)}"
                    alt={sample.filename}
                />
                {#if hasNext}
                    <button type="button" class="nav-btn nav-next" onclick={next}>&#8250;</button>
                {/if}
            </div>

            <div class="lightbox-footer">
                <span>{currentIndex + 1} / {samples.length}</span>
                <span>{new Date(sample.modified * 1000).toLocaleString()}</span>
            </div>
        </div>
    </div>
{/if}

<style>
    .lightbox-overlay {
        position: fixed; inset: 0;
        background: rgba(0, 0, 0, 0.85);
        display: flex; align-items: center; justify-content: center;
        z-index: 1000;
    }
    .lightbox {
        display: flex; flex-direction: column;
        max-width: 90vw; max-height: 90vh;
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }
    .lightbox-header {
        display: flex; align-items: center; justify-content: space-between;
        padding: 12px 16px;
        border-bottom: 1px solid var(--border-subtle);
    }
    .lightbox-filename { font-family: var(--font-mono); font-size: 12px; color: var(--text-primary); }
    .lightbox-close { background: none !important; border: none !important; color: var(--text-muted); font-size: 18px; cursor: pointer; padding: 0 !important; }
    .lightbox-close:hover { color: var(--text-primary); }
    .lightbox-image {
        position: relative;
        display: flex; align-items: center; justify-content: center;
        min-height: 300px;
        flex: 1;
        padding: 16px;
    }
    .lightbox-image img { max-width: 100%; max-height: 70vh; object-fit: contain; border-radius: var(--radius); }
    .nav-btn {
        position: absolute; top: 50%; transform: translateY(-50%);
        background: var(--bg-tertiary) !important; border: 1px solid var(--border) !important;
        color: var(--text-primary); font-size: 24px; padding: 8px 12px !important;
        cursor: pointer; border-radius: var(--radius);
        z-index: 1;
    }
    .nav-btn:hover { background: var(--bg-elevated) !important; }
    .nav-prev { left: 16px; }
    .nav-next { right: 16px; }
    .lightbox-footer {
        display: flex; align-items: center; justify-content: space-between;
        padding: 10px 16px;
        border-top: 1px solid var(--border-subtle);
        font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);
    }
</style>
