<script lang="ts">
    import type { Snippet } from 'svelte';

    interface Props {
        label: string;
        description?: string;
        error?: string;
        children: Snippet;
    }
    let { label, description = '', error = '', children }: Props = $props();
</script>

<div class="form-field" class:has-error={!!error}>
    <span class="field-label">{label}</span>
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
    .field-label { font-family: var(--font-mono); font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-primary); }
    .field-desc { font-size: 11px; color: var(--text-muted); line-height: 1.4; }
    .field-error { font-size: 11px; color: var(--error); }
    .has-error :global(input), .has-error :global(select) { border-color: var(--error) !important; }
</style>
