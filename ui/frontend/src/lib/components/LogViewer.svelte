<script lang="ts">
	import { trainingState } from '$lib/stores/training';
	import { tick } from 'svelte';

	let logs = $derived($trainingState.logs);
	let container: HTMLDivElement;
	let autoScroll = $state(true);

	$effect(() => {
		if (logs.length && autoScroll && container) {
			tick().then(() => {
				container.scrollTop = container.scrollHeight;
			});
		}
	});

	function handleScroll() {
		if (!container) return;
		const atBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 40;
		autoScroll = atBottom;
	}

	function levelClass(level: string): string {
		switch (level) {
			case 'ERROR': return 'log-error';
			case 'WARNING': return 'log-warning';
			case 'DEBUG': return 'log-debug';
			default: return 'log-info';
		}
	}

	function formatTime(ts: number): string {
		const d = new Date(ts * 1000);
		return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
	}
</script>

<div class="log-viewer" bind:this={container} onscroll={handleScroll}>
	{#if logs.length === 0}
		<div class="log-empty">No log output yet.</div>
	{:else}
		{#each logs as entry, i (i)}
			<div class="log-line {levelClass(entry.level)}">
				<span class="log-time">{formatTime(entry.timestamp)}</span>
				<span class="log-level">{entry.level.padEnd(7)}</span>
				<span class="log-msg">{entry.message}</span>
			</div>
		{/each}
	{/if}
</div>

<style>
	.log-viewer {
		flex: 1;
		overflow-y: auto;
		padding: 8px 0;
		font-family: var(--font-mono);
		font-size: 11.5px;
		line-height: 1.7;
		min-height: 0;
	}

	.log-empty {
		padding: 24px 16px;
		color: var(--text-muted);
		font-style: italic;
		font-family: var(--font-sans);
		font-size: 13px;
	}

	.log-line {
		display: flex;
		gap: 12px;
		padding: 1px 16px;
		white-space: pre-wrap;
		word-break: break-all;
	}

	.log-line:hover {
		background: rgba(255, 255, 255, 0.02);
	}

	.log-time {
		color: var(--text-muted);
		flex-shrink: 0;
	}

	.log-level {
		flex-shrink: 0;
		width: 56px;
	}

	.log-msg {
		flex: 1;
		color: var(--text-secondary);
	}

	.log-info .log-level { color: var(--text-muted); }
	.log-info .log-msg { color: var(--text-secondary); }

	.log-warning .log-level { color: var(--warning); }
	.log-warning .log-msg { color: var(--warning); }

	.log-error .log-level { color: var(--error); font-weight: 600; }
	.log-error .log-msg { color: var(--error); }
	.log-error { background: var(--error-dim); }

	.log-debug .log-level { color: var(--text-muted); }
	.log-debug .log-msg { color: var(--text-muted); }
</style>
