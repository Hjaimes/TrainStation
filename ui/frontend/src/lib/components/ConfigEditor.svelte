<script lang="ts">
	import { configState, validateConfig } from '$lib/stores/config';

	let configText = $state('{}');
	let state = $derived($configState);

	async function handleValidate() {
		try {
			const config = JSON.parse(configText);
			configState.update(s => ({ ...s, config }));
			await validateConfig(config);
		} catch (e) {
			configState.update(s => ({
				...s,
				valid: false,
				errors: [(e as Error).message],
				warnings: []
			}));
		}
	}
</script>

<div class="config-editor">
	<textarea
		bind:value={configText}
		rows="8"
		placeholder='Paste YAML or JSON config here...'
		spellcheck="false"
	></textarea>

	<div class="editor-footer">
		<button onclick={handleValidate}>Validate</button>
		{#if state.valid === true}
			<span class="validation-ok">✓ Valid</span>
		{:else if state.valid === false}
			<span class="validation-err">✗ Invalid</span>
		{/if}
	</div>

	{#if state.errors.length > 0}
		<div class="error-list">
			{#each state.errors as err}
				<div class="error-item">• {err}</div>
			{/each}
		</div>
	{/if}

	{#if state.warnings.length > 0}
		<div class="warning-list">
			{#each state.warnings as warn}
				<div class="warning-item">• {warn}</div>
			{/each}
		</div>
	{/if}
</div>

<style>
	.config-editor {
		padding: 16px;
		display: flex;
		flex-direction: column;
		gap: 10px;
	}

	.editor-footer {
		display: flex;
		align-items: center;
		gap: 12px;
	}

	.validation-ok {
		font-family: var(--font-mono);
		font-size: 12px;
		color: var(--success);
	}

	.validation-err {
		font-family: var(--font-mono);
		font-size: 12px;
		color: var(--error);
	}

	.error-list, .warning-list {
		font-family: var(--font-mono);
		font-size: 11px;
		padding: 8px 12px;
		border-radius: var(--radius);
	}

	.error-list {
		background: var(--error-dim);
		color: var(--error);
	}

	.warning-list {
		background: rgba(245, 158, 11, 0.1);
		color: var(--warning);
	}

	.error-item, .warning-item {
		padding: 2px 0;
	}
</style>
