<script lang="ts">
    interface SelectOption {
        value: string;
        label: string;
    }
    interface Props {
        value: string;
        options: SelectOption[];
        disabled?: boolean;
        onchange?: (value: string) => void;
    }
    let { value = $bindable(), options, disabled = false, onchange }: Props = $props();

    function handleChange(e: Event) {
        value = (e.target as HTMLSelectElement).value;
        onchange?.(value);
    }
</script>

<select class="select-input" {value} {disabled} onchange={handleChange}>
    {#each options as opt}
        <option value={opt.value} selected={opt.value === value}>{opt.label}</option>
    {/each}
</select>

<style>
    .select-input {
        font-family: var(--font-mono);
        font-size: 13px;
        background: var(--bg-primary);
        color: var(--text-secondary);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 8px 10px;
        width: 100%;
        cursor: pointer;
        transition: border-color 0.15s ease;
        appearance: none;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath d='M3 5l3 3 3-3' stroke='%23997c87' fill='none' stroke-width='1.5'/%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-position: right 10px center;
        padding-right: 30px;
    }
    .select-input:focus { outline: none; border-color: var(--accent-dim); }
    .select-input:disabled { opacity: 0.4; cursor: not-allowed; }
    .select-input option { background: var(--bg-secondary); color: var(--text-primary); }
</style>
