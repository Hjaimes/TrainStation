<script lang="ts">
	import { onMount } from 'svelte';
	import { configState, loadDefaults, validateConfig } from '$lib/stores/config';
	import { queueStore } from '$lib/stores/queue';
	import { goto } from '$app/navigation';
	import SectionTabs from '$lib/components/configure/SectionTabs.svelte';
	import ConfigureToolbar from '$lib/components/configure/ConfigureToolbar.svelte';
	import PresetDropdown from '$lib/components/configure/PresetDropdown.svelte';
	import SavePresetDialog from '$lib/components/configure/SavePresetDialog.svelte';
	import PreflightModal from '$lib/components/configure/PreflightModal.svelte';
	import ModelSection from '$lib/components/configure/ModelSection.svelte';
	import NetworkSection from '$lib/components/configure/NetworkSection.svelte';
	import DataSection from '$lib/components/configure/DataSection.svelte';
	import TrainingSection from '$lib/components/configure/TrainingSection.svelte';
	import SamplingSection from '$lib/components/configure/SamplingSection.svelte';
	import OutputSection from '$lib/components/configure/OutputSection.svelte';
	import RawConfigEditor from '$lib/components/configure/RawConfigEditor.svelte';

	let activeSection = $state('model');
	let showSavePreset = $state(false);
	let showPreflight = $state(false);
	let presetDropdown: PresetDropdown;
	let cfg = $derived($configState);

	onMount(async () => {
		if (!cfg.config) {
			await loadDefaults('wan');
		}
	});

	async function handleValidate() {
		if (cfg.config) await validateConfig(cfg.config);
	}

	function handleStart() {
		if (cfg.config) {
			showPreflight = true;
		}
	}

	async function handleQueue() {
		if (!cfg.config) return;
		await handleValidate();
		if (!cfg.valid) return;
		const arch = cfg.config.model?.architecture ?? 'unknown';
		const name = `${arch}-${new Date().toISOString().slice(0, 16).replace('T', '-')}`;
		await queueStore.addJob(name, cfg.config as unknown as Record<string, unknown>);
		await goto('/queue');
	}
</script>

<div class="configure-page">
	<ConfigureToolbar
		onvalidate={handleValidate}
		onstart={handleStart}
		onqueue={handleQueue}
		valid={cfg.valid}
	>
		{#snippet preset()}
			<PresetDropdown bind:this={presetDropdown} />
			<button type="button" class="btn-save-preset" onclick={() => showSavePreset = true}>Save As...</button>
		{/snippet}
	</ConfigureToolbar>

	<SavePresetDialog
		open={showSavePreset}
		onclose={() => showSavePreset = false}
		onsaved={() => presetDropdown?.refresh()}
	/>

	<PreflightModal
		open={showPreflight}
		onclose={() => showPreflight = false}
	/>

	<div class="configure-body">
		<SectionTabs active={activeSection} onselect={(s) => activeSection = s} />

		<div class="section-content">
			{#if activeSection === 'model'}
				<ModelSection />
			{:else if activeSection === 'network'}
				<NetworkSection />
			{:else if activeSection === 'data'}
				<DataSection />
			{:else if activeSection === 'training'}
				<TrainingSection />
			{:else if activeSection === 'sampling'}
				<SamplingSection />
			{:else if activeSection === 'output'}
				<OutputSection />
			{:else if activeSection === 'raw'}
				<RawConfigEditor />
			{/if}

			{#if cfg.errors.length > 0}
				<div class="error-list">
					{#each cfg.errors as err}
						<div class="error-item">{'\u2022'} {err}</div>
					{/each}
				</div>
			{/if}
			{#if cfg.warnings.length > 0}
				<div class="warning-list">
					{#each cfg.warnings as warn}
						<div class="warning-item">{'\u2022'} {warn}</div>
					{/each}
				</div>
			{/if}
		</div>
	</div>
</div>

<style>
	.configure-page {
		display: flex;
		flex-direction: column;
		height: 100%;
		gap: 16px;
	}
	.configure-body {
		display: flex;
		flex: 1;
		min-height: 0;
		border: 1px solid var(--border-subtle);
		border-radius: var(--radius);
		overflow: hidden;
	}
	.section-content {
		flex: 1;
		overflow-y: auto;
		padding: 24px;
	}
	.error-list, .warning-list {
		font-family: var(--font-mono);
		font-size: 11px;
		padding: 10px 14px;
		border-radius: var(--radius);
		margin-top: 16px;
	}
	.error-list { background: rgba(239, 68, 68, 0.1); color: var(--error); }
	.warning-list { background: rgba(245, 158, 11, 0.1); color: var(--warning); }
	.error-item, .warning-item { padding: 2px 0; }
	:global(.btn-save-preset) {
		font-family: var(--font-mono);
		font-size: 12px;
		background: var(--bg-tertiary);
		color: var(--text-secondary);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		padding: 6px 12px;
		cursor: pointer;
	}
	:global(.btn-save-preset:hover) {
		color: var(--text-primary);
		border-color: var(--accent-dim);
	}
</style>
