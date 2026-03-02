<script lang="ts">
    interface Props {
        value: string;
        placeholder?: string;
        disabled?: boolean;
        onchange?: (value: string) => void;
    }
    let { value = $bindable(), placeholder = 'Enter path...', disabled = false, onchange }: Props = $props();

    function handleInput(e: Event) {
        value = (e.target as HTMLInputElement).value;
        onchange?.(value);
    }
</script>

<div class="path-input-wrapper">
    <span class="path-icon">&#128193;</span>
    <input type="text" class="path-input" {value} {placeholder} {disabled} oninput={handleInput} spellcheck="false" />
</div>

<style>
    .path-input-wrapper {
        display: flex;
        align-items: center;
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        transition: border-color 0.15s ease;
    }
    .path-input-wrapper:focus-within { border-color: var(--accent-dim); }
    .path-icon { padding: 0 4px 0 10px; font-size: 13px; flex-shrink: 0; line-height: 1; }
    .path-input {
        font-family: var(--font-mono);
        font-size: 13px;
        background: transparent;
        color: var(--text-primary);
        border: none;
        padding: 8px 10px 8px 4px;
        width: 100%;
    }
    .path-input:focus { outline: none; }
    .path-input:disabled { opacity: 0.4; cursor: not-allowed; }
    .path-input::placeholder { color: var(--text-muted); }
</style>
