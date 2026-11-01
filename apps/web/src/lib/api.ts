const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export type Health = {
  status: string;
  providers: Record<string, { status: string; latency_ms?: number }>;
};

export async function getHealth(signal?: AbortSignal): Promise<Health> {
  const res = await fetch(`${API_URL}/health`, { signal });
  if (!res.ok) throw new Error(`health ${res.status}`);
  return res.json();
}
