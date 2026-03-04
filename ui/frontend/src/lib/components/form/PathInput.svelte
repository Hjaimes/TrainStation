<script lang="ts">
    interface Props {
        value: string;
        placeholder?: string;
        disabled?: boolean;
        mode?: 'directory' | 'file';
        extensions?: string[];
        allowHuggingFace?: boolean;
        onchange?: (value: string) => void;
    }
    let {
        value = $bindable(),
        placeholder = 'Enter path...',
        disabled = false,
        mode = 'directory',
        extensions,
        allowHuggingFace = false,
        onchange,
    }: Props = $props();

    let browsing = $state(false);

    // Detect HuggingFace model IDs (e.g. "Wan-AI/Wan2.1-T2V-14B")
    let isHfId = $derived(() => {
        if (!allowHuggingFace || !value?.trim()) return false;
        const v = value.trim();
        if (v.startsWith('/') || v.startsWith('\\') || v.startsWith('./') || v.startsWith('../') || v.startsWith('~')) return false;
        if (v.length >= 2 && v[1] === ':' && /[a-zA-Z]/.test(v[0])) return false;
        if (v.includes('\\')) return false;
        return /^[a-zA-Z0-9_.-]+\/[a-zA-Z0-9_.-]+(?:\/[a-zA-Z0-9_.-]+)*$/.test(v);
    });

    function handleInput(e: Event) {
        value = (e.target as HTMLInputElement).value;
        onchange?.(value);
    }

    async function browse() {
        if (disabled || browsing) return;
        browsing = true;
        try {
            const endpoint = mode === 'file' ? '/api/browse/file' : '/api/browse/directory';
            const body: Record<string, unknown> = {};

            // Use current value's parent dir as starting point
            if (value && !isHfId()) {
                const parts = value.replace(/\\/g, '/').split('/');
                if (mode === 'file') {
                    parts.pop();
                }
                const dir = parts.join('/');
                if (dir) body.initial_dir = dir;
            }

            if (mode === 'file' && extensions?.length) {
                body.extensions = extensions;
            }

            const resp = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });

            if (!resp.ok) return;
            const data = await resp.json();
            if (!data.cancelled && data.path) {
                value = data.path;
                onchange?.(value);
            }
        } catch {
            // Dialog failed silently - user can still type manually
        } finally {
            browsing = false;
        }
    }
</script>

<div class="path-input-wrapper">
    <button
        type="button"
        class="browse-btn"
        title={mode === 'file' ? 'Browse for file' : 'Browse for folder'}
        disabled={disabled || browsing}
        onclick={browse}
    >
        {#if browsing}
            <span class="spinner"></span>
        {:else}
            &#128193;
        {/if}
    </button>
    <input type="text" class="path-input" {value} {placeholder} {disabled} oninput={handleInput} spellcheck="false" />
    {#if isHfId()}
        <span class="hf-badge" title="HuggingFace model ID — will be downloaded automatically">HF</span>
    {/if}
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

    .browse-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        flex-shrink: 0;
        background: none !important;
        border: none !important;
        border-right: 1px solid var(--border-subtle) !important;
        border-radius: 0;
        padding: 0 !important;
        font-size: 13px;
        cursor: pointer;
        color: var(--text-muted);
        transition: color 0.12s ease, background 0.12s ease;
    }
    .browse-btn:hover:not(:disabled) {
        color: var(--accent);
        background: var(--accent-glow) !important;
    }
    .browse-btn:disabled {
        opacity: 0.4;
        cursor: not-allowed;
    }

    .spinner {
        display: inline-block;
        width: 12px;
        height: 12px;
        border: 2px solid var(--border-subtle);
        border-top-color: var(--accent);
        border-radius: 50%;
        animation: spin 0.6s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    .path-input {
        font-family: var(--font-mono);
        font-size: 13px;
        background: transparent;
        color: var(--text-secondary);
        border: none;
        padding: 8px 10px;
        width: 100%;
    }
    .path-input:focus { outline: none; }
    .path-input:disabled { opacity: 0.4; cursor: not-allowed; }
    .path-input::placeholder { color: var(--text-muted); }

    .hf-badge {
        flex-shrink: 0;
        font-family: var(--font-mono);
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 0.05em;
        color: var(--accent);
        background: var(--accent-glow);
        border: 1px solid var(--accent-dim);
        border-radius: 3px;
        padding: 2px 5px;
        margin-right: 8px;
        cursor: default;
    }
</style>
