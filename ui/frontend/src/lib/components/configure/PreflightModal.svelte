<script lang="ts">
	import { configState } from '$lib/stores/config';
	import { startTraining } from '$lib/stores/training';
	import { goto } from '$app/navigation';

	let storeState = $derived($configState);

	interface Check {
		name: string;
		status: 'ok' | 'warning' | 'error';
		message: string;
	}

	interface PreflightResult {
		checks: Check[];
		can_start: boolean;
	}

	interface Props {
		open: boolean;
		onclose: () => void;
	}
	let { open, onclose }: Props = $props();

	let loading = $state(true);
	let checks = $state<Check[]>([]);
	let canStart = $state(false);
	let errorMessage = $state('');
	let starting = $state(false);

	$effect(() => {
		if (open) {
			runChecks();
		}
	});

	async function runChecks() {
		loading = true;
		checks = [];
		canStart = false;
		errorMessage = '';

		const config = storeState.config;
		if (!config) {
			errorMessage = 'No configuration loaded';
			loading = false;
			return;
		}

		try {
			const resp = await fetch('/api/preflight/check', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ config }),
			});
			if (!resp.ok) {
				errorMessage = `Server error: ${resp.status}`;
				loading = false;
				return;
			}
			const result: PreflightResult = await resp.json();
			checks = result.checks;
			canStart = result.can_start;
		} catch (e) {
			errorMessage = (e as Error).message;
		} finally {
			loading = false;
		}
	}

	async function handleStart() {
		const config = storeState.config;
		if (!config) return;
		starting = true;
		try {
			await startTraining(config);
			onclose();
			await goto('/training');
		} catch (e) {
			errorMessage = (e as Error).message;
		} finally {
			starting = false;
		}
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Escape') onclose();
	}

	function statusIcon(status: string): string {
		if (status === 'ok') return '\u2713';
		if (status === 'warning') return '\u26A0';
		return '\u2717';
	}
</script>

{#if open}
<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="modal-overlay" onclick={onclose} onkeydown={handleKeydown}>
	<!-- svelte-ignore a11y_click_events_have_key_events -->
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div class="modal" onclick={(e) => e.stopPropagation()}>
		<h3 class="modal-title">Pre-flight Check</h3>

		<div class="modal-body">
			{#if loading}
				<div class="loading-state">
					<span class="spinner"></span>
					<span>Checking configuration...</span>
				</div>
			{:else if errorMessage}
				<div class="error-banner">{errorMessage}</div>
			{:else}
				<div class="check-list">
					{#each checks as check}
						<div class="check-item">
							<span class="check-icon status-{check.status}">{statusIcon(check.status)}</span>
							<span class="check-name">{check.name}</span>
							<span class="check-message status-{check.status}">{check.message}</span>
						</div>
					{/each}
				</div>
			{/if}
		</div>

		<div class="modal-actions">
			<button type="button" onclick={onclose}>Cancel</button>
			{#if !loading}
				<button type="button" class="btn-recheck" onclick={runChecks}>Re-check</button>
			{/if}
			<button
				type="button"
				class="btn-start"
				onclick={handleStart}
				disabled={!canStart || loading || starting}
			>
				{starting ? 'Starting...' : '\u25B6 Start Training'}
			</button>
		</div>
	</div>
</div>
{/if}

<style>
	.modal-overlay {
		position: fixed; inset: 0;
		background: rgba(0, 0, 0, 0.6);
		display: flex; align-items: center; justify-content: center;
		z-index: 1000;
	}
	.modal {
		background: var(--bg-secondary);
		border: 1px solid var(--border);
		border-radius: 8px;
		padding: 24px;
		min-width: 480px;
		max-width: 600px;
	}
	.modal-title {
		font-family: var(--font-mono);
		font-size: 14px;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--text-primary);
		margin-bottom: 16px;
	}
	.modal-body {
		display: flex;
		flex-direction: column;
		gap: 8px;
		margin-bottom: 20px;
		min-height: 80px;
	}
	.loading-state {
		display: flex;
		align-items: center;
		gap: 10px;
		font-family: var(--font-mono);
		font-size: 13px;
		color: var(--text-secondary);
		padding: 12px 0;
	}
	.spinner {
		display: inline-block;
		width: 16px;
		height: 16px;
		border: 2px solid var(--border);
		border-top-color: var(--accent);
		border-radius: 50%;
		animation: spin 0.6s linear infinite;
	}
	@keyframes spin {
		to { transform: rotate(360deg); }
	}
	.error-banner {
		font-family: var(--font-mono);
		font-size: 12px;
		color: var(--error);
		background: rgba(239, 68, 68, 0.1);
		padding: 10px 14px;
		border-radius: var(--radius);
	}
	.check-list {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.check-item {
		display: flex;
		align-items: baseline;
		gap: 8px;
		font-family: var(--font-mono);
		font-size: 12px;
		padding: 6px 8px;
		background: var(--bg-primary);
		border-radius: var(--radius);
	}
	.check-icon {
		flex-shrink: 0;
		width: 18px;
		text-align: center;
		font-weight: 700;
	}
	.check-name {
		flex-shrink: 0;
		color: var(--text-primary);
		font-weight: 600;
		min-width: 120px;
	}
	.check-message {
		color: var(--text-secondary);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.status-ok { color: var(--success); }
	.status-warning { color: var(--warning); }
	.status-error { color: var(--error); }
	.modal-actions {
		display: flex;
		justify-content: flex-end;
		gap: 8px;
	}
	.btn-recheck {
		opacity: 0.8;
	}
	.btn-start {
		background: var(--accent) !important;
		border-color: var(--accent) !important;
		color: #fff !important;
	}
	.btn-start:hover:not(:disabled) {
		background: var(--accent-dim) !important;
	}
	.btn-start:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}
</style>
