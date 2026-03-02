<script lang="ts">
	import TrainingControls from '$lib/components/TrainingControls.svelte';
	import MetricsDisplay from '$lib/components/MetricsDisplay.svelte';
	import LogViewer from '$lib/components/LogViewer.svelte';
	import LossChart from '$lib/components/monitor/LossChart.svelte';
	import SamplePreview from '$lib/components/monitor/SamplePreview.svelte';
</script>

<div class="monitor-page">
	<div class="page-header">
		<h1>Monitor</h1>
		<TrainingControls />
	</div>

	<div class="dashboard">
		<div class="panel chart-panel">
			<div class="panel-header">Loss</div>
			<LossChart />
		</div>

		<div class="panel metrics-panel">
			<div class="panel-header">Metrics</div>
			<MetricsDisplay />
		</div>

		<div class="panel log-panel">
			<div class="panel-header">Log Output</div>
			<LogViewer />
		</div>

		<div class="panel sample-panel">
			<div class="panel-header">Latest Sample</div>
			<SamplePreview />
		</div>
	</div>
</div>

<style>
	.monitor-page { display: flex; flex-direction: column; height: 100%; gap: 20px; }
	.page-header { display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
	h1 { font-family: var(--font-mono); font-size: 18px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--text-primary); }

	.dashboard {
		display: grid;
		grid-template-columns: 1fr 320px;
		grid-template-rows: 1fr 1fr;
		gap: 16px;
		flex: 1;
		min-height: 0;
	}
	.panel {
		background: var(--bg-secondary);
		border: 1px solid var(--border-subtle);
		border-radius: var(--radius);
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}
	.panel-header {
		font-family: var(--font-mono);
		font-size: 11px;
		font-weight: 600;
		letter-spacing: 0.1em;
		text-transform: uppercase;
		color: var(--text-muted);
		padding: 12px 16px;
		border-bottom: 1px solid var(--border-subtle);
		flex-shrink: 0;
	}
	.chart-panel { grid-column: 1; grid-row: 1; }
	.metrics-panel { grid-column: 2; grid-row: 1; }
	.log-panel { grid-column: 1; grid-row: 2; }
	.sample-panel { grid-column: 2; grid-row: 2; }
</style>
