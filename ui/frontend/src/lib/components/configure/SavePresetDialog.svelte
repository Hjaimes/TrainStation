<script lang="ts">
    import { get } from 'svelte/store';
    import { configState } from '$lib/stores/config';

    interface Props {
        open: boolean;
        onclose: () => void;
        onsaved?: () => void;
    }
    let { open, onclose, onsaved }: Props = $props();

    let name = $state('');
    let saving = $state(false);
    let error = $state('');

    async function save() {
        const config = get(configState).config;

        if (!name.trim()) {
            error = 'Name is required';
            return;
        }

        saving = true;
        error = '';
        try {
            const resp = await fetch('/api/presets/user', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name.trim(), config }),
            });
            if (!resp.ok) {
                const data = await resp.json();
                error = data.detail || 'Failed to save';
                return;
            }
            name = '';
            onsaved?.();
            onclose();
        } catch (e) {
            error = (e as Error).message;
        } finally {
            saving = false;
        }
    }

    function handleKeydown(e: KeyboardEvent) {
        if (e.key === 'Escape') onclose();
        if (e.key === 'Enter') save();
    }
</script>

{#if open}
<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="modal-overlay" onclick={onclose} onkeydown={handleKeydown}>
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div class="modal" onclick={(e) => e.stopPropagation()}>
        <h3 class="modal-title">Save Preset</h3>
        <div class="modal-body">
            <label class="input-label" for="preset-name-input">Preset Name</label>
            <input
                id="preset-name-input"
                type="text"
                class="preset-name-input"
                bind:value={name}
                placeholder="my-preset-name"
            />
            {#if error}
                <span class="save-error">{error}</span>
            {/if}
        </div>
        <div class="modal-actions">
            <button type="button" onclick={onclose}>Cancel</button>
            <button type="button" class="btn-save" onclick={save} disabled={saving}>
                {saving ? 'Saving...' : 'Save'}
            </button>
        </div>
    </div>
</div>
{/if}

<style>
    .modal-overlay {
        position: fixed; inset: 0;
        background: rgba(0, 0, 0, 0.6);
        display: flex; align-items: center; justify-content: center;
        z-index: 1000;
    }
    .modal {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 24px;
        min-width: 400px;
        max-width: 480px;
    }
    .modal-title {
        font-family: var(--font-mono);
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-primary);
        margin-bottom: 16px;
    }
    .modal-body { display: flex; flex-direction: column; gap: 8px; margin-bottom: 20px; }
    .input-label {
        font-family: var(--font-mono);
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-secondary);
    }
    .preset-name-input {
        font-family: var(--font-mono);
        font-size: 13px;
        background: var(--bg-primary);
        color: var(--text-primary);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 8px 10px;
        width: 100%;
    }
    .preset-name-input:focus { outline: none; border-color: var(--accent-dim); }
    .save-error { font-family: var(--font-mono); font-size: 11px; color: var(--error); }
    .modal-actions { display: flex; justify-content: flex-end; gap: 8px; }
    .btn-save { background: var(--accent) !important; border-color: var(--accent) !important; color: #fff !important; }
    .btn-save:hover { background: var(--accent-dim) !important; }
</style>
