import type { TimeseriesResponse } from "../types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const INJECT_URL = import.meta.env.VITE_INJECT_URL ?? "";

export class ApiError extends Error {}

export async function askAgent(prompt: string, topK: number): Promise<{ answer: string; raw: unknown }> {
  const resp = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, top_k: topK }),
  });
  const data = await resp.json().catch(() => null);
  if (!resp.ok) {
    throw new ApiError(data?.detail ?? `Agent request failed (${resp.status})`);
  }
  return { answer: data?.answer ?? "No answer returned.", raw: data };
}

export async function fetchTimeseries(start: Date, end: Date): Promise<TimeseriesResponse> {
  const params = new URLSearchParams({ start: start.toISOString(), end: end.toISOString() });
  const resp = await fetch(`${API_BASE}/transactions/timeseries?${params}`);
  if (!resp.ok) {
    throw new ApiError(`Failed to fetch transactions (${resp.status})`);
  }
  return resp.json();
}

export function injectConfigured(): boolean {
  return INJECT_URL.length > 0;
}

export async function injectTransactions(durationSeconds: number): Promise<void> {
  if (!INJECT_URL) {
    throw new ApiError("No producer configured (VITE_INJECT_URL is unset).");
  }
  const url = `${INJECT_URL}?duration_seconds=${Math.trunc(durationSeconds)}`;
  const resp = await fetch(url, { method: "POST" });
  if (!resp.ok) {
    throw new ApiError(`Inject failed (${resp.status})`);
  }
}
