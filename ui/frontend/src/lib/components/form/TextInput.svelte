<script lang="ts">
    interface Props {
        value: string;
        placeholder?: string;
        disabled?: boolean;
        onchange?: (value: string) => void;
    }
    let { value = $bindable(), placeholder = '', disabled = false, onchange }: Props = $props();

    function handleInput(e: Event) {
        value = (e.target as HTMLInputElement).value;
        onchange?.(value);
    }
</script>

<input type="text" class="text-input" {value} {placeholder} {disabled} oninput={handleInput} />

<style>
    .text-input {
        font-family: var(--font-mono);
        font-size: 13px;
        background: var(--bg-primary);
        color: var(--text-secondary);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 8px 10px;
        width: 100%;
        transition: border-color 0.15s ease;
    }
    .text-input:focus { outline: none; border-color: var(--accent-dim); }
    .text-input:disabled { opacity: 0.4; cursor: not-allowed; }
    .text-input::placeholder { color: var(--text-muted); }
</style>
