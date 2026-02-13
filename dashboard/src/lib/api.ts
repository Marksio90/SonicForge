const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

async function fetchApi<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}/api/v1${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!response.ok) {
    throw new Error(`API Error: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

// Pipeline
export const runPipeline = (genre?: string, energy?: number) =>
  fetchApi("/pipeline/run", {
    method: "POST",
    body: JSON.stringify({ genre, energy }),
  });

export const runPipelineSync = (genre?: string) =>
  fetchApi(`/pipeline/run-sync?${genre ? `genre=${genre}` : ""}`  , { method: "POST" });

export const generateBatch = (count: number, genre?: string) =>
  fetchApi(`/pipeline/batch?count=${count}${genre ? `&genre=${genre}` : ""}`, { method: "POST" });

// Tracks
export const createConcept = (data: { genre?: string; energy?: number; mood?: string }) =>
  fetchApi("/tracks/concept", { method: "POST", body: JSON.stringify(data) });

export const evaluateTrack = (trackId: string, genre?: string) =>
  fetchApi(`/tracks/evaluate/${trackId}?${genre ? `genre=${genre}` : ""}`, { method: "POST" });

// Stream
export const getStreamStatus = () => fetchApi("/stream/status");
export const controlStream = (action: string) =>
  fetchApi("/stream/control", { method: "POST", body: JSON.stringify({ action }) });

export const getStreamHealth = () => fetchApi("/stream/health");
export const getQueue = () => fetchApi("/stream/queue");

export const submitListenerRequest = (data: { request_type: string; value: string }) =>
  fetchApi("/stream/request", { method: "POST", body: JSON.stringify(data) });

// Schedule
export const scheduleNext = () => fetchApi("/schedule/next", { method: "POST" });
export const checkBuffer = () => fetchApi("/schedule/buffer");

// Analytics
export const getAnalyticsSnapshot = () => fetchApi("/analytics/snapshot");
export const getGenrePerformance = () => fetchApi("/analytics/genre-performance");
export const getDailyReport = () => fetchApi("/analytics/daily-report");

// Agents
export const getAgentStatuses = () => fetchApi("/agents/status");
export const getAgentActivity = (limit = 50) => fetchApi(`/agents/activity?limit=${limit}`);

// Dashboard
export const getDashboardOverview = () => fetchApi("/dashboard/overview");

// Genres
export const getGenres = () => fetchApi("/genres");

// Visuals
export const generateVisual = (trackId: string, genre?: string) =>
  fetchApi(`/visuals/generate/${trackId}?${genre ? `genre=${genre}` : ""}`, { method: "POST" });

// WebSocket
export function connectDashboardWS(onMessage: (data: any) => void): WebSocket {
  const ws = new WebSocket(`${WS_URL}/ws/dashboard`);
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    onMessage(data);
  };
  return ws;
}
