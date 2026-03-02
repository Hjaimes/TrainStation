<script lang="ts">
    interface Sample {
        filename: string;
        absolute_path: string;
        size: number;
        modified: number;
    }

    interface Props {
        samples: Sample[];
        onselect: (sample: Sample) => void;
    }
    let { samples, onselect }: Props = $props();

    function formatTime(ts: number): string {
        return new Date(ts * 1000).toLocaleString();
    }

    function formatSize(bytes: number): string {
        if (bytes < 1024) return `${bytes}B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
    }
</script>

<div class="sample-grid">
    {#each samples as sample}
        <!-- svelte-ignore a11y_click_events_have_key_events -->
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <div class="sample-card" onclick={() => onselect(sample)}>
            <div class="sample-thumb">
                <img
                    src="/api/samples/file?path={encodeURIComponent(sample.absolute_path)}"
                    alt={sample.filename}
                    loading="lazy"
                />
            </div>
            <div class="sample-meta">
                <span class="sample-name" title={sample.filename}>{sample.filename}</span>
                <span class="sample-date">{formatTime(sample.modified)} · {formatSize(sample.size)}</span>
            </div>
        </div>
    {/each}
</div>

<style>
    .sample-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 16px;
    }
    .sample-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius);
        overflow: hidden;
        cursor: pointer;
        transition: border-color 0.15s ease;
    }
    .sample-card:hover { border-color: var(--accent-dim); }
    .sample-thumb {
        aspect-ratio: 1;
        overflow: hidden;
        background: var(--bg-primary);
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .sample-thumb img { width: 100%; height: 100%; object-fit: cover; }
    .sample-meta { padding: 10px 12px; }
    .sample-name {
        display: block;
        font-family: var(--font-mono);
        font-size: 11px;
        color: var(--text-primary);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .sample-date {
        display: block;
        font-size: 10px;
        color: var(--text-muted);
        margin-top: 4px;
    }
</style>
