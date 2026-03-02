import { writable } from 'svelte/store';
import type { TrainConfig } from '$lib/types/config';

export interface ConfigState {
	config: TrainConfig | null;
	dirty: boolean;
	valid: boolean | null;
	errors: string[];
	warnings: string[];
}

function createConfigStore() {
	const { subscribe, update, set } = writable<ConfigState>({
		config: null,
		dirty: false,
		valid: null,
		errors: [],
		warnings: []
	});

	return {
		subscribe,
		update,
		set,
		/** Update a nested config section. Marks store as dirty. */
		updateSection<K extends keyof TrainConfig>(section: K, values: Partial<TrainConfig[K]>) {
			update(s => {
				if (!s.config) return s;
				const currentSection = s.config[section];
				// Handle null sections (like network)
				if (currentSection === null || currentSection === undefined) {
					return {
						...s,
						dirty: true,
						config: {
							...s.config,
							[section]: values as TrainConfig[K]
						}
					};
				}
				return {
					...s,
					dirty: true,
					config: {
						...s.config,
						[section]: { ...(currentSection as object), ...values }
					}
				};
			});
		},
		/** Set an entire section at once */
		setSection<K extends keyof TrainConfig>(section: K, value: TrainConfig[K]) {
			update(s => {
				if (!s.config) return s;
				return {
					...s,
					dirty: true,
					config: { ...s.config, [section]: value }
				};
			});
		}
	};
}

export const configState = createConfigStore();

export async function loadDefaults(arch: string): Promise<TrainConfig> {
	const resp = await fetch(`/api/config/defaults/${arch}`);
	const data: TrainConfig = await resp.json();
	configState.set({ config: data, dirty: false, valid: null, errors: [], warnings: [] });
	return data;
}

export async function validateConfig(config: TrainConfig) {
	const resp = await fetch('/api/config/validate', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ config })
	});
	const result = await resp.json();
	configState.update(s => ({
		...s,
		valid: result.valid,
		errors: result.errors,
		warnings: result.warnings
	}));
	return result;
}
