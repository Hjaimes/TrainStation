<script lang="ts">
    interface Props {
        value: boolean;
        disabled?: boolean;
        onchange?: (value: boolean) => void;
    }
    let { value = $bindable(), disabled = false, onchange }: Props = $props();

    function toggle() {
        if (disabled) return;
        value = !value;
        onchange?.(value);
    }
</script>

<button class="toggle" class:on={value} class:disabled {disabled} onclick={toggle} type="button" role="switch" aria-checked={value}>
    <span class="toggle-thumb"></span>
</button>

<style>
    .toggle {
        position: relative;
        width: 36px;
        height: 20px;
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: 10px;
        cursor: pointer;
        padding: 0;
        transition: all 0.2s ease;
        flex-shrink: 0;
    }
    .toggle:hover:not(.disabled) { border-color: var(--accent-dim); }
    .toggle.on { background: var(--accent); border-color: var(--accent); }
    .toggle.disabled { opacity: 0.4; cursor: not-allowed; }
    .toggle-thumb {
        position: absolute;
        top: 2px;
        left: 2px;
        width: 14px;
        height: 14px;
        background: var(--text-muted);
        border-radius: 50%;
        transition: all 0.2s ease;
    }
    .toggle.on .toggle-thumb { left: 18px; background: #fff; }
</style>
