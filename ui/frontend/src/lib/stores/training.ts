import { writable } from 'svelte/store';
import type { TrainConfig } from '$lib/types/config';

export interface LogEntry {
	level: string;
	message: string;
	timestamp: number;
}

export interface TrainingState {
	connected: boolean;
	alive: boolean;
	step: number;
	totalSteps: number;
	loss: number;
	avgLoss: number;
	lr: number;
	epoch: number;
	logs: LogEntry[];
	lastError: string | null;
	lossHistory: { step: number; loss: number; avgLoss: number }[];
	vramPeakMb: number;
	etaSeconds: number;
	startTime: number;
	latestSample: { path: string; prompt: string; step: number } | null;
}

const MAX_LOGS = 500;

function createTrainingStore() {
	const { subscribe, update, set } = writable<TrainingState>({
		connected: false,
		alive: false,
		step: 0,
		totalSteps: 0,
		loss: 0,
		avgLoss: 0,
		lr: 0,
		epoch: 0,
		logs: [],
		lastError: null,
		lossHistory: [],
		vramPeakMb: 0,
		etaSeconds: 0,
		startTime: 0,
		latestSample: null,
	});

	let ws: WebSocket | null = null;
	let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

	function connectWebSocket() {
		if (ws && ws.readyState <= WebSocket.OPEN) return;

		const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
		const url = `${protocol}//${window.location.host}/ws/training`;
		ws = new WebSocket(url);

		ws.onopen = () => {
			update(s => ({ ...s, connected: true }));
		};

		ws.onclose = () => {
			update(s => ({ ...s, connected: false }));
			ws = null;
			reconnectTimer = setTimeout(connectWebSocket, 2000);
		};

		ws.onerror = () => {
			ws?.close();
		};

		ws.onmessage = (event) => {
			const data = JSON.parse(event.data);
			handleEvent(data);
		};
	}

	function handleEvent(event: Record<string, unknown>) {
		update(s => {
			const next = { ...s };

			switch (event.type) {
				case 'TrainingStartedEvent':
					next.alive = true;
					next.totalSteps = event.total_steps as number;
					next.step = 0;
					next.loss = 0;
					next.avgLoss = 0;
					next.lastError = null;
					next.lossHistory = [];
					next.startTime = Date.now() / 1000;
					next.latestSample = null;
					next.logs = [...s.logs, {
						level: 'INFO',
						message: `Training started: ${event.architecture} (${event.method})`,
						timestamp: event.timestamp as number
					}].slice(-MAX_LOGS);
					break;

				case 'StepEvent':
					next.step = event.step as number;
					next.totalSteps = event.total_steps as number;
					next.loss = event.loss as number;
					next.avgLoss = event.avg_loss as number;
					next.lr = event.lr as number;
					next.epoch = event.epoch as number;
					next.lossHistory = [...s.lossHistory, {
						step: event.step as number,
						loss: event.loss as number,
						avgLoss: event.avg_loss as number,
					}];
					// Cap at 10000 points
					if (next.lossHistory.length > 10000) {
						next.lossHistory = next.lossHistory.slice(-10000);
					}
					// Compute ETA from elapsed time
					if (s.startTime > 0) {
						const elapsed = (Date.now() / 1000) - s.startTime;
						const stepsPerSec = next.step / elapsed;
						next.etaSeconds = stepsPerSec > 0 ? (next.totalSteps - next.step) / stepsPerSec : 0;
					}
					// Capture VRAM if present
					if (event.vram_peak_mb !== undefined) {
						next.vramPeakMb = event.vram_peak_mb as number;
					}
					break;

				case 'LogEvent':
					next.logs = [...s.logs, {
						level: event.level as string,
						message: event.message as string,
						timestamp: event.timestamp as number
					}].slice(-MAX_LOGS);
					break;

				case 'ErrorEvent':
					next.lastError = event.message as string;
					if (event.is_fatal) next.alive = false;
					next.logs = [...s.logs, {
						level: 'ERROR',
						message: event.message as string,
						timestamp: event.timestamp as number
					}].slice(-MAX_LOGS);
					break;

				case 'TrainingCompleteEvent':
					next.alive = false;
					next.step = event.final_step as number;
					next.loss = event.final_loss as number;
					next.logs = [...s.logs, {
						level: 'INFO',
						message: `Training complete. Final loss: ${(event.final_loss as number).toFixed(4)}`,
						timestamp: event.timestamp as number
					}].slice(-MAX_LOGS);
					break;

				case 'EpochEvent':
					next.epoch = event.epoch as number;
					break;

				case 'CheckpointEvent':
					next.logs = [...s.logs, {
						level: 'INFO',
						message: `Checkpoint saved: ${event.path}`,
						timestamp: event.timestamp as number
					}].slice(-MAX_LOGS);
					break;

				case 'SampleEvent':
					next.latestSample = {
						path: event.path as string,
						prompt: event.prompt as string,
						step: event.step as number,
					};
					next.logs = [...s.logs, {
						level: 'INFO',
						message: `Sample generated at step ${event.step}`,
						timestamp: event.timestamp as number,
					}].slice(-MAX_LOGS);
					break;
			}

			return next;
		});
	}

	function disconnect() {
		if (reconnectTimer) clearTimeout(reconnectTimer);
		ws?.close();
		ws = null;
	}

	return { subscribe, connectWebSocket, disconnect };
}

export const trainingState = createTrainingStore();

export async function startTraining(config: TrainConfig | null, mode = 'train') {
	const resp = await fetch('/api/training/start', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ config, mode })
	});
	return resp.json();
}

export async function stopTraining() {
	const resp = await fetch('/api/training/stop', { method: 'POST' });
	return resp.json();
}

export async function pauseTraining() {
	const resp = await fetch('/api/training/pause', { method: 'POST' });
	return resp.json();
}

export async function resumeTraining() {
	const resp = await fetch('/api/training/resume', { method: 'POST' });
	return resp.json();
}
