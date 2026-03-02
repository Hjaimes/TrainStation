<script lang="ts">
	import { trainingState } from '$lib/stores/training';

	let sample = $derived($trainingState.latestSample);
</script>

<div class="sample-preview">
	{#if sample}
		<div class="sample-image">
			<img src="/api/samples/file?path={encodeURIComponent(sample.path)}" alt="Step {sample.step}" />
		</div>
		<div class="sample-info">
			<span class="sample-step">Step {sample.step}</span>
			<span class="sample-prompt" title={sample.prompt}>{sample.prompt}</span>
		</div>
		<a href="/samples" class="sample-link">View all samples &rarr;</a>
	{:else}
		<div class="no-sample">
			<span class="no-sample-icon">&#9714;</span>
			<span class="no-sample-text">No samples generated yet</span>
		</div>
	{/if}
</div>

<style>
	.sample-preview {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		padding: 16px;
		height: 100%;
		gap: 12px;
	}
	.sample-image {
		flex: 1;
		display: flex;
		align-items: center;
		justify-content: center;
		min-height: 0;
		width: 100%;
	}
	.sample-image img {
		max-width: 100%;
		max-height: 100%;
		object-fit: contain;
		border-radius: var(--radius);
		border: 1px solid var(--border-subtle);
	}
	.sample-info {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 4px;
		width: 100%;
	}
	.sample-step {
		font-family: var(--font-mono);
		font-size: 11px;
		font-weight: 600;
		color: var(--accent);
		text-transform: uppercase;
		letter-spacing: 0.08em;
	}
	.sample-prompt {
		font-size: 12px;
		color: var(--text-muted);
		text-align: center;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		max-width: 100%;
	}
	.sample-link {
		font-family: var(--font-mono);
		font-size: 11px;
		color: var(--accent);
		text-decoration: none;
	}
	.sample-link:hover { text-decoration: underline; }
	.no-sample {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 8px;
		color: var(--text-muted);
	}
	.no-sample-icon { font-size: 32px; opacity: 0.3; }
	.no-sample-text { font-size: 13px; font-style: italic; }
</style>
