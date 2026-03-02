<script lang="ts">
	import { trainingState } from '$lib/stores/training';

	let state = $derived($trainingState);
	let progress = $derived(state.totalSteps > 0 ? (state.step / state.totalSteps) * 100 : 0);

	function formatEta(seconds: number): string {
		if (seconds <= 0) return '\u2014';
		const h = Math.floor(seconds / 3600);
		const m = Math.floor((seconds % 3600) / 60);
		const s = Math.floor(seconds % 60);
		if (h > 0) return `${h}h ${m}m`;
		if (m > 0) return `${m}m ${s}s`;
		return `${s}s`;
	}
</script>

<div class="metrics">
	<div class="metric-grid">
		<div class="metric">
			<span class="metric-label">Step</span>
			<span class="metric-value">{state.step}<span class="metric-dim">/{state.totalSteps}</span></span>
		</div>
		<div class="metric">
			<span class="metric-label">Loss</span>
			<span class="metric-value">{state.loss.toFixed(4)}</span>
		</div>
		<div class="metric">
			<span class="metric-label">Avg Loss</span>
			<span class="metric-value">{state.avgLoss.toFixed(4)}</span>
		</div>
		<div class="metric">
			<span class="metric-label">LR</span>
			<span class="metric-value">{state.lr.toExponential(2)}</span>
		</div>
		<div class="metric">
			<span class="metric-label">Epoch</span>
			<span class="metric-value">{state.epoch}</span>
		</div>
		<div class="metric">
			<span class="metric-label">Status</span>
			<span class="metric-value status" class:alive={state.alive} class:error={state.lastError}>
				{state.alive ? 'Running' : state.lastError ? 'Error' : 'Idle'}
			</span>
		</div>
		<div class="metric">
			<span class="metric-label">ETA</span>
			<span class="metric-value">{formatEta(state.etaSeconds)}</span>
		</div>
		<div class="metric">
			<span class="metric-label">VRAM Peak</span>
			<span class="metric-value">{state.vramPeakMb > 0 ? state.vramPeakMb.toFixed(0) + ' MB' : '\u2014'}</span>
		</div>
	</div>

	<div class="progress-container">
		<div class="progress-bar" style="width: {progress}%"></div>
		<span class="progress-label">{progress.toFixed(1)}%</span>
	</div>
</div>

<style>
	.metrics {
		padding: 16px;
	}

	.metric-grid {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		gap: 14px;
		margin-bottom: 16px;
	}

	.metric {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.metric-label {
		font-family: var(--font-mono);
		font-size: 10px;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.1em;
		color: var(--text-muted);
	}

	.metric-value {
		font-family: var(--font-mono);
		font-size: 16px;
		font-weight: 600;
		color: var(--text-primary);
		letter-spacing: -0.02em;
	}

	.metric-dim {
		color: var(--text-muted);
		font-size: 13px;
		font-weight: 400;
	}

	.status.alive {
		color: var(--success);
	}

	.status.error {
		color: var(--error);
	}

	.progress-container {
		position: relative;
		height: 6px;
		background: var(--bg-primary);
		border-radius: 3px;
		overflow: visible;
	}

	.progress-bar {
		height: 100%;
		background: var(--accent);
		border-radius: 3px;
		transition: width 0.3s ease;
		box-shadow: 0 0 8px var(--accent-glow);
	}

	.progress-label {
		position: absolute;
		right: 0;
		top: -20px;
		font-family: var(--font-mono);
		font-size: 10px;
		color: var(--text-muted);
		letter-spacing: 0.05em;
	}
</style>
