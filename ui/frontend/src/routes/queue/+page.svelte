<script lang="ts">
	import { onMount } from 'svelte';
	import { queueStore } from '$lib/stores/queue';
	import type { Job } from '$lib/stores/queue';

	let jobs = $derived($queueStore);

	onMount(() => {
		queueStore.refresh();
		const interval = setInterval(() => queueStore.refresh(), 5000);
		return () => clearInterval(interval);
	});

	const queuedJobs = $derived(jobs.filter(j => j.status === 'queued'));
	const runningJobs = $derived(jobs.filter(j => j.status === 'running'));
	const completedJobs = $derived(jobs.filter(j => j.status === 'completed' || j.status === 'failed'));

	function formatTime(ts: number | null): string {
		if (!ts) return '\u2014';
		return new Date(ts * 1000).toLocaleString();
	}

	function getArch(job: Job): string {
		return (job.config?.model as Record<string, unknown>)?.architecture as string ?? 'unknown';
	}

	function getMethod(job: Job): string {
		return (job.config?.training as Record<string, unknown>)?.method as string ?? 'lora';
	}
</script>

<div class="queue-page">
	<div class="page-header">
		<h1>Queue</h1>
		<button type="button" onclick={() => queueStore.refresh()}>Refresh</button>
	</div>

	{#if jobs.length === 0}
		<div class="empty-state">
			<span class="empty-icon">{'\u2630'}</span>
			<p>No jobs in queue. Add a job from the Configure page.</p>
		</div>
	{:else}
		{#if runningJobs.length > 0}
			<section class="job-section">
				<h2 class="section-label">Running</h2>
				{#each runningJobs as job (job.id)}
					<div class="job-card running">
						<div class="job-main">
							<span class="job-name">{job.name}</span>
							<span class="job-badge badge-running">Running</span>
							<span class="job-meta">{getArch(job)} · {getMethod(job)}</span>
						</div>
						<div class="job-actions">
							<a href="/monitor" class="btn-monitor">View Monitor →</a>
						</div>
					</div>
				{/each}
			</section>
		{/if}

		{#if queuedJobs.length > 0}
			<section class="job-section">
				<h2 class="section-label">Queued ({queuedJobs.length})</h2>
				{#each queuedJobs as job, i (job.id)}
					<div class="job-card queued">
						<div class="job-main">
							<span class="job-position">#{i + 1}</span>
							<span class="job-name">{job.name}</span>
							<span class="job-badge badge-queued">Queued</span>
							<span class="job-meta">{getArch(job)} · {getMethod(job)}</span>
						</div>
						<div class="job-actions">
							{#if i > 0}
								<button type="button" onclick={() => queueStore.reorderJob(job.id, i - 1)}>{'\u2191'}</button>
							{/if}
							{#if i < queuedJobs.length - 1}
								<button type="button" onclick={() => queueStore.reorderJob(job.id, i + 1)}>{'\u2193'}</button>
							{/if}
							<button type="button" class="btn-clone" onclick={() => queueStore.cloneJob(job.id)}>Clone</button>
							<button type="button" class="btn-delete" onclick={() => queueStore.removeJob(job.id)}>{'\u2715'}</button>
						</div>
					</div>
				{/each}
			</section>
		{/if}

		{#if completedJobs.length > 0}
			<section class="job-section">
				<h2 class="section-label">Completed ({completedJobs.length})</h2>
				{#each completedJobs as job (job.id)}
					<div class="job-card completed" class:failed={job.status === 'failed'}>
						<div class="job-main">
							<span class="job-name">{job.name}</span>
							<span class="job-badge" class:badge-completed={job.status === 'completed'} class:badge-failed={job.status === 'failed'}>
								{job.status === 'completed' ? 'Done' : 'Failed'}
							</span>
							<span class="job-meta">{getArch(job)} · {formatTime(job.completed_at)}</span>
						</div>
						<div class="job-actions">
							<button type="button" onclick={() => queueStore.cloneJob(job.id)}>Clone</button>
							<button type="button" class="btn-delete" onclick={() => queueStore.removeJob(job.id)}>{'\u2715'}</button>
						</div>
					</div>
				{/each}
			</section>
		{/if}
	{/if}
</div>

<style>
	.queue-page { display: flex; flex-direction: column; height: 100%; gap: 24px; }
	.page-header { display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
	h1 { font-family: var(--font-mono); font-size: 18px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--text-primary); }

	.empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; flex: 1; gap: 12px; color: var(--text-muted); }
	.empty-icon { font-size: 48px; opacity: 0.2; }
	.empty-state p { font-size: 14px; font-style: italic; }

	.job-section { display: flex; flex-direction: column; gap: 8px; }
	.section-label { font-family: var(--font-mono); font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); }

	.job-card {
		display: flex; align-items: center; justify-content: space-between;
		padding: 14px 16px;
		background: var(--bg-secondary);
		border: 1px solid var(--border-subtle);
		border-radius: var(--radius);
	}
	.job-card.running { border-left: 3px solid var(--accent); }
	.job-card.failed { border-left: 3px solid var(--error); }

	.job-main { display: flex; align-items: center; gap: 12px; }
	.job-position { font-family: var(--font-mono); font-size: 12px; color: var(--text-muted); min-width: 24px; }
	.job-name { font-weight: 600; font-size: 14px; color: var(--text-primary); }
	.job-meta { font-size: 12px; color: var(--text-muted); }

	.job-badge {
		font-family: var(--font-mono); font-size: 10px; font-weight: 600;
		text-transform: uppercase; letter-spacing: 0.06em;
		padding: 2px 8px; border-radius: 2px;
	}
	.badge-running { background: var(--accent-glow); color: var(--accent); }
	.badge-queued { background: var(--bg-tertiary); color: var(--text-muted); }
	.badge-completed { background: rgba(52, 211, 153, 0.1); color: var(--success); }
	.badge-failed { background: rgba(239, 68, 68, 0.1); color: var(--error); }

	.job-actions { display: flex; gap: 6px; }
	.btn-delete { color: var(--error) !important; border-color: var(--error) !important; }
	.btn-delete:hover { background: rgba(239, 68, 68, 0.1) !important; }
	.btn-clone { color: var(--text-secondary); }
	.btn-monitor { font-family: var(--font-mono); font-size: 12px; color: var(--accent); text-decoration: none; }
	.btn-monitor:hover { text-decoration: underline; }
</style>
