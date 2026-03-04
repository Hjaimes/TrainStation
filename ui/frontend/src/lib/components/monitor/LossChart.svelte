<script lang="ts">
	import { Chart, registerables } from 'chart.js';
	import { trainingState } from '$lib/stores/training';
	import { onMount, onDestroy } from 'svelte';

	Chart.register(...registerables);

	let canvas: HTMLCanvasElement;
	let chart: Chart | null = null;
	let history = $derived($trainingState.lossHistory);

	onMount(() => {
		chart = new Chart(canvas, {
			type: 'line',
			data: {
				labels: [],
				datasets: [
					{
						label: 'Loss',
						data: [],
						borderColor: 'rgba(245, 74, 177, 0.35)',
						borderWidth: 1,
						pointRadius: 0,
						tension: 0.1,
						fill: false,
					},
					{
						label: 'Avg Loss',
						data: [],
						borderColor: '#f53689',
						borderWidth: 2,
						pointRadius: 0,
						tension: 0.3,
						fill: false,
					},
				],
			},
			options: {
				responsive: true,
				maintainAspectRatio: false,
				animation: false,
				interaction: {
					mode: 'index',
					intersect: false,
				},
				scales: {
					x: {
						display: true,
						title: { display: true, text: 'Step', color: '#997c87', font: { family: 'JetBrains Mono', size: 10 } },
						grid: { color: 'rgba(255, 255, 255, 0.04)' },
						ticks: { color: '#997c87', font: { family: 'JetBrains Mono', size: 10 }, maxTicksLimit: 10 },
					},
					y: {
						display: true,
						title: { display: true, text: 'Loss', color: '#997c87', font: { family: 'JetBrains Mono', size: 10 } },
						grid: { color: 'rgba(255, 255, 255, 0.04)' },
						ticks: { color: '#997c87', font: { family: 'JetBrains Mono', size: 10 } },
					},
				},
				plugins: {
					legend: {
						display: true,
						position: 'top',
						align: 'end',
						labels: {
							color: '#c4a0ad',
							font: { family: 'JetBrains Mono', size: 10 },
							boxWidth: 12,
							boxHeight: 2,
							padding: 12,
						},
					},
					tooltip: {
						backgroundColor: '#1c181b',
						titleColor: '#ffdae8',
						bodyColor: '#c4a0ad',
						borderColor: '#36212d',
						borderWidth: 1,
						titleFont: { family: 'JetBrains Mono', size: 11 },
						bodyFont: { family: 'JetBrains Mono', size: 11 },
						padding: 8,
					},
				},
			},
		});
	});

	$effect(() => {
		if (chart && history) {
			chart.data.labels = history.map((h) => h.step);
			chart.data.datasets[0].data = history.map((h) => h.loss);
			chart.data.datasets[1].data = history.map((h) => h.avgLoss);
			chart.update('none'); // 'none' skips animation for performance
		}
	});

	onDestroy(() => {
		chart?.destroy();
		chart = null;
	});
</script>

<div class="chart-container">
	<canvas bind:this={canvas}></canvas>
</div>

<style>
	.chart-container {
		position: relative;
		width: 100%;
		height: 100%;
		min-height: 200px;
		padding: 12px;
	}
</style>
