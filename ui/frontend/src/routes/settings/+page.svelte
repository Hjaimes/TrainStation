<script lang="ts">
    import { FormField, SelectInput, NumberInput, TextInput } from '$lib/components/form';

    // Load settings from localStorage
    function loadSettings() {
        try {
            const saved = localStorage.getItem('trainer-settings');
            return saved ? JSON.parse(saved) : {};
        } catch {
            return {};
        }
    }

    function saveSettings() {
        localStorage.setItem('trainer-settings', JSON.stringify({
            defaultArchitecture,
            defaultOutputDir,
            maxLogEntries,
            reconnectInterval,
        }));
        showSaved = true;
        setTimeout(() => showSaved = false, 2000);
    }

    const saved = loadSettings();
    let defaultArchitecture = $state(saved.defaultArchitecture ?? 'wan');
    let defaultOutputDir = $state(saved.defaultOutputDir ?? './output');
    let maxLogEntries = $state(saved.maxLogEntries ?? 500);
    let reconnectInterval = $state(saved.reconnectInterval ?? 2000);
    let showSaved = $state(false);
</script>

<div class="settings-page">
    <div class="page-header">
        <h1>Settings</h1>
    </div>

    <div class="settings-panel">
        <h2 class="panel-title">Application Preferences</h2>
        <p class="panel-desc">These settings are stored locally in your browser.</p>

        <div class="settings-fields">
            <FormField label="Default Architecture" description="Architecture to load when opening Configure page">
                <SelectInput
                    value={defaultArchitecture}
                    options={[
                        { value: 'wan', label: 'Wan' },
                        { value: 'flux_2', label: 'Flux 2' },
                        { value: 'sd3', label: 'SD3' },
                        { value: 'sdxl', label: 'SDXL' },
                        { value: 'flux_1', label: 'Flux 1' },
                        { value: 'hunyuan_video', label: 'HunyuanVideo' },
                        { value: 'framepack', label: 'FramePack' },
                    ]}
                    onchange={(v) => defaultArchitecture = v}
                />
            </FormField>

            <FormField label="Default Output Directory" description="Default output path for new configs">
                <TextInput
                    value={defaultOutputDir}
                    onchange={(v) => defaultOutputDir = v}
                />
            </FormField>

            <FormField label="Max Log Entries" description="Maximum number of log lines to keep in memory">
                <NumberInput
                    value={maxLogEntries}
                    min={100}
                    max={10000}
                    step={100}
                    onchange={(v) => maxLogEntries = v}
                />
            </FormField>

            <FormField label="Reconnect Interval (ms)" description="WebSocket reconnect delay in milliseconds">
                <NumberInput
                    value={reconnectInterval}
                    min={500}
                    max={30000}
                    step={500}
                    onchange={(v) => reconnectInterval = v}
                />
            </FormField>
        </div>

        <div class="settings-actions">
            <button type="button" class="btn-save" onclick={saveSettings}>Save Settings</button>
            {#if showSaved}
                <span class="saved-badge">Saved!</span>
            {/if}
        </div>
    </div>
</div>

<style>
    .settings-page { display: flex; flex-direction: column; height: 100%; gap: 20px; }
    .page-header { flex-shrink: 0; }
    h1 { font-family: var(--font-mono); font-size: 18px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--text-primary); }

    .settings-panel {
        background: var(--bg-secondary);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius);
        padding: 24px;
        max-width: 600px;
    }
    .panel-title {
        font-family: var(--font-mono);
        font-size: 14px; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.06em;
        color: var(--text-primary);
        margin-bottom: 4px;
    }
    .panel-desc { font-size: 13px; color: var(--text-muted); margin-bottom: 20px; }
    .settings-fields { display: flex; flex-direction: column; gap: 16px; }
    .settings-actions { display: flex; align-items: center; gap: 12px; margin-top: 20px; }
    .btn-save { background: var(--accent) !important; border-color: var(--accent) !important; color: #fff !important; }
    .btn-save:hover { background: var(--accent-dim) !important; }
    .saved-badge { font-family: var(--font-mono); font-size: 12px; color: var(--success); }
</style>
