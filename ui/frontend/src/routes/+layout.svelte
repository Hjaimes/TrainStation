<script lang="ts">
	import '../app.css';
	import { onMount } from 'svelte';
	import { trainingState } from '$lib/stores/training';
	import ConnectionStatus from '$lib/components/ConnectionStatus.svelte';
	import { page } from '$app/state';

	onMount(() => {
		trainingState.connectWebSocket();
		return () => trainingState.disconnect();
	});

	const navItems = [
		{ label: 'Configure', href: '/configure', icon: '⚙' },
		{ label: 'Monitor', href: '/monitor', icon: '◈' },
		{ label: 'Samples', href: '/samples', icon: '◲' },
		{ label: 'Queue', href: '/queue', icon: '☰' },
		{ label: 'Settings', href: '/settings', icon: '⚡' },
	];
</script>

<div class="shell">
	<nav class="sidebar">
		<div class="sidebar-header">
			<span class="logo-mark">▲</span>
			<span class="logo-text">AI Trainer</span>
		</div>

		<div class="nav-items">
			{#each navItems as item}
				<a
					href={item.href}
					class="nav-item"
					class:active={page.url.pathname.startsWith(item.href)}
				>
					<span class="nav-icon">{item.icon}</span>
					<span class="nav-label">{item.label}</span>
				</a>
			{/each}
		</div>

		<div class="sidebar-footer">
			<ConnectionStatus />
		</div>
	</nav>

	<main class="content">
		<slot />
	</main>
</div>

<style>
	.shell {
		display: flex;
		height: 100vh;
		overflow: hidden;
	}

	.sidebar {
		width: var(--sidebar-width);
		min-width: var(--sidebar-width);
		background: var(--bg-secondary);
		border-right: 1px solid var(--border-subtle);
		display: flex;
		flex-direction: column;
		padding: 0;
	}

	.sidebar-header {
		padding: 20px 18px;
		display: flex;
		align-items: center;
		gap: 10px;
		border-bottom: 1px solid var(--border-subtle);
	}

	.logo-mark {
		font-size: 18px;
		color: var(--accent);
		line-height: 1;
	}

	.logo-text {
		font-family: var(--font-mono);
		font-weight: 700;
		font-size: 13px;
		letter-spacing: 0.06em;
		text-transform: uppercase;
		color: var(--text-primary);
	}

	.nav-items {
		flex: 1;
		padding: 12px 10px;
		display: flex;
		flex-direction: column;
		gap: 2px;
	}

	.nav-item {
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 10px 12px;
		border-radius: var(--radius);
		text-decoration: none;
		color: var(--text-secondary);
		font-size: 13px;
		font-weight: 500;
		transition: all 0.12s ease;
		position: relative;
	}

	.nav-item:hover {
		background: var(--bg-tertiary);
		color: var(--text-primary);
	}

	.nav-item.active {
		background: var(--accent-glow);
		color: var(--accent);
		border-left: 2px solid var(--accent);
		padding-left: 10px;
	}

	.nav-icon {
		font-size: 14px;
		width: 20px;
		text-align: center;
		flex-shrink: 0;
	}

	.nav-label {
		flex: 1;
	}

	.sidebar-footer {
		padding: 14px 18px;
		border-top: 1px solid var(--border-subtle);
	}

	.content {
		flex: 1;
		overflow-y: auto;
		padding: 24px 28px;
	}
</style>
