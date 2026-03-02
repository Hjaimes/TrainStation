<script lang="ts">
    interface Props {
        active: string;
        onselect: (section: string) => void;
    }
    let { active, onselect }: Props = $props();

    const sections = [
        { id: 'model', label: 'Model', icon: '\u25C6' },
        { id: 'network', label: 'Network', icon: '\u25C7' },
        { id: 'data', label: 'Data', icon: '\u25A4' },
        { id: 'training', label: 'Training', icon: '\u26A1' },
        { id: 'sampling', label: 'Sampling', icon: '\u25F2' },
        { id: 'output', label: 'Output', icon: '\u2197' },
        { id: 'raw', label: 'Raw Config', icon: '{ }' },
    ];
</script>

<div class="section-tabs">
    {#each sections as sec}
        <button
            class="tab"
            class:active={active === sec.id}
            onclick={() => onselect(sec.id)}
            type="button"
        >
            <span class="tab-icon">{sec.icon}</span>
            <span class="tab-label">{sec.label}</span>
        </button>
    {/each}
</div>

<style>
    .section-tabs {
        display: flex;
        flex-direction: column;
        gap: 2px;
        min-width: 160px;
        padding: 8px;
        border-right: 1px solid var(--border-subtle);
        background: var(--bg-secondary);
    }
    .tab {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border: none;
        border-radius: var(--radius);
        background: transparent;
        color: var(--text-secondary);
        font-family: var(--font-sans);
        font-size: 13px;
        font-weight: 500;
        text-transform: none;
        letter-spacing: normal;
        cursor: pointer;
        transition: all 0.12s ease;
        text-align: left;
    }
    .tab:hover { background: var(--bg-tertiary); color: var(--text-primary); }
    .tab.active { background: var(--accent-glow); color: var(--accent); }
    .tab-icon { font-size: 13px; width: 18px; text-align: center; flex-shrink: 0; }
    .tab-label { flex: 1; }
</style>
