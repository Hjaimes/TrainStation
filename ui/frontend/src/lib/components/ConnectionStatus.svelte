<script lang="ts">
	import { trainingState } from '$lib/stores/training';

	let connected = $derived($trainingState.connected);
</script>

<div class="connection-status">
	<span class="dot" class:connected></span>
	<span class="label">{connected ? 'Connected' : 'Disconnected'}</span>
</div>

<style>
	.connection-status {
		display: flex;
		align-items: center;
		gap: 8px;
		font-family: var(--font-mono);
		font-size: 11px;
		color: var(--text-muted);
	}

	.dot {
		width: 7px;
		height: 7px;
		border-radius: 50%;
		background: var(--error);
		flex-shrink: 0;
		transition: background 0.2s ease;
		box-shadow: 0 0 6px var(--error);
	}

	.dot.connected {
		background: var(--success);
		box-shadow: 0 0 6px var(--success);
	}

	.label {
		text-transform: uppercase;
		letter-spacing: 0.06em;
	}
</style>
