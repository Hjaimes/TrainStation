<script lang="ts">
    interface Props {
        value: number;
        min?: number;
        max?: number;
        step?: number | string;
        placeholder?: string;
        disabled?: boolean;
        onchange?: (value: number) => void;
    }
    let { value = $bindable(), min, max, step = 1, placeholder = '', disabled = false, onchange }: Props = $props();

    function handleInput(e: Event) {
        const v = parseFloat((e.target as HTMLInputElement).value);
        if (!isNaN(v)) {
            value = v;
            onchange?.(v);
        }
    }
</script>

<input type="number" class="num-input" {value} {min} {max} {step} {placeholder} {disabled} oninput={handleInput} />

<style>
    .num-input {
        font-family: var(--font-mono);
        font-size: 13px;
        background: var(--bg-primary);
        color: var(--text-primary);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 8px 10px;
        width: 100%;
        transition: border-color 0.15s ease;
    }
    .num-input:focus { outline: none; border-color: var(--accent-dim); }
    .num-input:disabled { opacity: 0.4; cursor: not-allowed; }
    .num-input::placeholder { color: var(--text-muted); }
    /* Hide spin buttons */
    .num-input::-webkit-inner-spin-button,
    .num-input::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }
    .num-input { -moz-appearance: textfield; }
</style>
