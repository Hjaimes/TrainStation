<script lang="ts">
    import type { Snippet } from 'svelte';

    interface Props {
        onvalidate: () => void;
        onstart: () => void;
        onqueue: () => void;
        valid: boolean | null;
        preset?: Snippet;
    }
    let { onvalidate, onstart, onqueue, valid, preset }: Props = $props();
</script>

<div class="toolbar">
    <div class="toolbar-left">
        {#if preset}
            {@render preset()}
        {/if}
    </div>
    <div class="toolbar-right">
        <button class="btn-validate" onclick={onvalidate} type="button">
            Validate
        </button>
        {#if valid === true}
            <span class="valid-badge">&#x2713; Valid</span>
        {:else if valid === false}
            <span class="invalid-badge">&#x2717; Invalid</span>
        {/if}
        <button class="btn-primary" onclick={onstart} type="button">
            &#x25B6; Start Training
        </button>
        <button class="btn-queue" onclick={onqueue} type="button">
            + Queue
        </button>
    </div>
</div>

<style>
    .toolbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 16px;
        background: var(--bg-secondary);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius);
        flex-shrink: 0;
    }
    .toolbar-left { display: flex; align-items: center; gap: 12px; }
    .toolbar-right { display: flex; align-items: center; gap: 8px; }
    .btn-primary {
        background: var(--accent) !important;
        border-color: var(--accent) !important;
        color: #fff !important;
    }
    .btn-primary:hover { background: var(--accent-dim) !important; }
    .btn-queue { opacity: 0.8; }
    .valid-badge { font-family: var(--font-mono); font-size: 12px; color: var(--success); }
    .invalid-badge { font-family: var(--font-mono); font-size: 12px; color: var(--error); }
</style>
