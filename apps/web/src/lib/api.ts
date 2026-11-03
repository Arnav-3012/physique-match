import type { AnalyzeRequest, AnalyzeResponse, PlanRequest, PlanResponse } from "./types.gen";

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

/** A rejected request. `message` is safe to show: the API's 422 details carry field paths, never the input. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly retryAfter?: number,
  ) {
    super(message);
  }
}

async function post<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (e) {
    if (signal?.aborted) throw e;
    throw new ApiError(0, "The server could not be reached.");
  }
  if (!res.ok) {
    let detail = "";
    try {
      const j = await res.json();
      detail = typeof j.detail === "string" ? j.detail : "The request was not accepted.";
    } catch {
      /* no body */
    }
    const retry = Number(res.headers.get("Retry-After")) || undefined;
    throw new ApiError(res.status, detail || `The server answered ${res.status}.`, retry);
  }
  return res.json();
}

export const postAnalyze = (req: AnalyzeRequest, signal?: AbortSignal) => post<AnalyzeResponse>("/analyze", req, signal);
export const postPlan = (req: PlanRequest, signal?: AbortSignal) => post<PlanResponse>("/plan", req, signal);
