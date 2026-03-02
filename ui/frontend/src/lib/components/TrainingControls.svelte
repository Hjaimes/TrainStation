<script lang="ts">
	import { trainingState, startTraining, stopTraining, pauseTraining, resumeTraining } from '$lib/stores/training';
	import { configState } from '$lib/stores/config';

	let alive = $derived($trainingState.alive);

	async function handleStart() {
		const config = $configState.config;
		await startTraining(config);
	}
</script>

<div class="controls">
	{#if !alive}
		<button class="btn-start" onclick={handleStart}>
			▶ Start
		</button>
	{:else}
		<button class="btn-stop" onclick={stopTraining}>
			■ Stop
		</button>
		<button onclick={pauseTraining}>
			❚❚ Pause
		</button>
		<button onclick={resumeTraining}>
			▶ Resume
		</button>
	{/if}
</div>

<style>
	.controls {
		display: flex;
		gap: 8px;
	}

	.btn-start {
		background: var(--accent);
		border-color: var(--accent);
		color: #fff;
	}

	.btn-start:hover {
		background: var(--accent-dim);
		border-color: var(--accent);
	}

	.btn-stop {
		background: var(--error-dim);
		border-color: var(--error);
		color: var(--error);
	}

	.btn-stop:hover {
		background: rgba(239, 68, 68, 0.25);
	}
</style>
