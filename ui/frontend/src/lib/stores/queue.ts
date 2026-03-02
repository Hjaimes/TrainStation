import { writable } from 'svelte/store';

export interface Job {
    id: string;
    name: string;
    status: string;
    config: Record<string, unknown>;
    created_at: number;
    started_at: number | null;
    completed_at: number | null;
    result: unknown;
}

function createQueueStore() {
    const { subscribe, set } = writable<Job[]>([]);

    async function refresh() {
        try {
            const resp = await fetch('/api/queue');
            const jobs: Job[] = await resp.json();
            set(jobs);
        } catch {
            set([]);
        }
    }

    async function addJob(name: string, config: Record<string, unknown>) {
        await fetch('/api/queue/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, config }),
        });
        await refresh();
    }

    async function removeJob(id: string) {
        await fetch(`/api/queue/${id}`, { method: 'DELETE' });
        await refresh();
    }

    async function cloneJob(id: string) {
        await fetch(`/api/queue/${id}/clone`, { method: 'POST' });
        await refresh();
    }

    async function reorderJob(id: string, newIndex: number) {
        await fetch(`/api/queue/${id}/reorder`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ index: newIndex }),
        });
        await refresh();
    }

    return { subscribe, refresh, addJob, removeJob, cloneJob, reorderJob };
}

export const queueStore = createQueueStore();
