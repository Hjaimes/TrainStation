<script lang="ts">
    import { configState } from '$lib/stores/config';

    let state = $derived($configState);
    let rawText = $state('');
    let parseError = $state('');
    let syncing = false;

    // Update rawText when config changes (unless user is editing)
    $effect(() => {
        if (state.config && !syncing) {
            rawText = JSON.stringify(state.config, null, 2);
        }
    });

    function handleBlur() {
        syncing = true;
        try {
            const parsed = JSON.parse(rawText);
            configState.update(s => ({ ...s, config: parsed, dirty: true }));
            parseError = '';
        } catch (e) {
            parseError = (e as Error).message;
        }
        // Allow effect to run again after a tick
        setTimeout(() => { syncing = false; }, 0);
    }
</script>

<div class="raw-editor">
    <div class="raw-header">
        <span class="raw-label">JSON Configuration</span>
        {#if parseError}
            <span class="parse-error">Parse error: {parseError}</span>
        {/if}
    </div>
    <textarea bind:value={rawText} onblur={handleBlur} rows="30" spellcheck="false"></textarea>
</div>

<style>
    .raw-editor {
        display: flex;
        flex-direction: column;
        gap: 8px;
        height: 100%;
    }
    .raw-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .raw-label {
        font-family: var(--font-mono);
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
    }
    .parse-error {
        font-family: var(--font-mono);
        font-size: 11px;
        color: var(--error);
    }
    textarea {
        flex: 1;
        min-height: 400px;
        font-family: var(--font-mono);
        font-size: 12px;
        background: var(--bg-primary);
        color: var(--text-primary);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 12px;
        resize: vertical;
        width: 100%;
        line-height: 1.6;
    }
    textarea:focus {
        outline: none;
        border-color: var(--accent-dim);
    }
</style>
