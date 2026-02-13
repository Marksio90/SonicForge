"use client";

import { useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface DashboardData {
  stream_status: Record<string, string>;
  current_track: Record<string, string> | null;
  queue_length: number;
  agents: Array<{ agent: string; status: string; timestamp?: string }>;
  recent_tracks: Array<Record<string, any>>;
  analytics: Record<string, any>;
}

export default function DashboardPage() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchDashboard() {
      try {
        const res = await fetch(`${API_URL}/api/v1/dashboard/overview`);
        if (res.ok) {
          setData(await res.json());
        }
      } catch {
        // API not available yet
      } finally {
        setLoading(false);
      }
    }
    fetchDashboard();
    const interval = setInterval(fetchDashboard, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Dashboard</h1>
          <p className="text-gray-500 text-sm mt-1">
            SonicForge AI Radio — Control Center
          </p>
        </div>
        <StreamIndicator isLive={data?.stream_status?.active === "true"} />
      </header>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Stream Status"
          value={data?.stream_status?.active === "true" ? "LIVE" : "OFFLINE"}
          color={data?.stream_status?.active === "true" ? "green" : "red"}
          icon="📡"
        />
        <StatCard
          title="Queue Buffer"
          value={`${data?.queue_length ?? 0} tracks`}
          color={
            (data?.queue_length ?? 0) > 10
              ? "green"
              : (data?.queue_length ?? 0) > 5
                ? "yellow"
                : "red"
          }
          icon="🎵"
        />
        <StatCard
          title="Concurrent Viewers"
          value={data?.analytics?.concurrent_viewers ?? "—"}
          color="cyan"
          icon="👥"
        />
        <StatCard
          title="Active Agents"
          value={`${data?.agents?.filter((a) => a.status !== "stopped").length ?? 0}/7`}
          color="accent"
          icon="🤖"
        />
      </div>

      {/* Current Track */}
      <div className="bg-sf-surface rounded-xl border border-sf-border p-6">
        <h2 className="text-lg font-semibold text-white mb-4">
          Now Playing
        </h2>
        {data?.current_track && Object.keys(data.current_track).length > 0 ? (
          <div className="flex items-center gap-6">
            <div className="w-16 h-16 rounded-lg bg-sf-accent/20 flex items-center justify-center text-2xl">
              🎵
            </div>
            <div>
              <p className="text-xl text-white font-medium">
                {data.current_track.title || "SonicForge Radio"}
              </p>
              <p className="text-gray-400 text-sm">
                {data.current_track.genre || "Electronic"} •{" "}
                {data.current_track.bpm ? `${data.current_track.bpm} BPM` : ""}
              </p>
            </div>
            <div className="ml-auto flex gap-3">
              <ActionButton label="Skip" icon="⏭" action="skip" />
              <ActionButton label="Restart Stream" icon="🔄" action="restart" />
            </div>
          </div>
        ) : (
          <p className="text-gray-500">No track currently playing</p>
        )}
      </div>

      {/* Agent Status Grid */}
      <div className="bg-sf-surface rounded-xl border border-sf-border p-6">
        <h2 className="text-lg font-semibold text-white mb-4">
          Agent Status
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
          {(data?.agents ?? defaultAgents).map((agent) => (
            <AgentCard key={agent.agent} agent={agent} />
          ))}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-sf-surface rounded-xl border border-sf-border p-6">
        <h2 className="text-lg font-semibold text-white mb-4">
          Quick Actions
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <PipelineButton label="Generate Track" genre={undefined} />
          <PipelineButton label="DnB Track" genre="drum_and_bass" />
          <PipelineButton label="Deep House" genre="house_deep" />
          <PipelineButton label="Trance Track" genre="trance_uplifting" />
          <PipelineButton label="Melodic Techno" genre="techno_melodic" />
          <PipelineButton label="Liquid DnB" genre="liquid_dnb" />
          <PipelineButton label="Ambient" genre="ambient" />
          <PipelineButton label="Batch (5 tracks)" genre="batch" />
        </div>
      </div>

      {/* Recent Activity */}
      <div className="bg-sf-surface rounded-xl border border-sf-border p-6">
        <h2 className="text-lg font-semibold text-white mb-4">
          Recent Activity
        </h2>
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {data?.recent_tracks && data.recent_tracks.length > 0 ? (
            data.recent_tracks.map((activity, i) => (
              <div
                key={i}
                className="flex items-center gap-3 py-2 px-3 rounded-lg hover:bg-sf-border/30"
              >
                <span className="text-xs text-gray-500 w-20 shrink-0">
                  {activity.timestamp
                    ? new Date(activity.timestamp).toLocaleTimeString()
                    : "—"}
                </span>
                <span className="text-sm text-gray-300">
                  [{activity.agent}] {activity.action}
                </span>
              </div>
            ))
          ) : (
            <p className="text-gray-500 text-sm">
              No recent activity. Start the pipeline to generate tracks!
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

// --- Sub Components ---

function StreamIndicator({ isLive }: { isLive: boolean }) {
  return (
    <div
      className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium ${
        isLive
          ? "bg-red-500/10 text-red-400 border border-red-500/30 glow-red"
          : "bg-gray-500/10 text-gray-400 border border-gray-500/30"
      }`}
    >
      <div
        className={`w-2 h-2 rounded-full ${
          isLive ? "bg-red-500 animate-pulse" : "bg-gray-500"
        }`}
      />
      {isLive ? "LIVE" : "OFFLINE"}
    </div>
  );
}

function StatCard({
  title,
  value,
  color,
  icon,
}: {
  title: string;
  value: string | number;
  color: string;
  icon: string;
}) {
  const colorMap: Record<string, string> = {
    green: "text-sf-green border-sf-green/30",
    red: "text-sf-red border-sf-red/30",
    yellow: "text-sf-yellow border-sf-yellow/30",
    cyan: "text-sf-cyan border-sf-cyan/30",
    accent: "text-sf-accent border-sf-accent/30",
  };

  return (
    <div
      className={`bg-sf-surface rounded-xl border p-5 ${colorMap[color] || "border-sf-border"}`}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-gray-500 uppercase tracking-wider">
          {title}
        </span>
        <span className="text-lg">{icon}</span>
      </div>
      <p className={`text-2xl font-bold ${colorMap[color]?.split(" ")[0] || "text-white"}`}>
        {value}
      </p>
    </div>
  );
}

function AgentCard({ agent }: { agent: { agent: string; status: string } }) {
  const statusColors: Record<string, string> = {
    idle: "bg-sf-green",
    working: "bg-sf-yellow animate-pulse",
    error: "bg-sf-red",
    stopped: "bg-gray-600",
  };

  const icons: Record<string, string> = {
    composer: "🎼",
    producer: "🎚️",
    critic: "🔍",
    scheduler: "📅",
    stream_master: "📡",
    analytics: "📊",
    visual: "🎨",
  };

  return (
    <div className="bg-sf-bg rounded-lg border border-sf-border p-3 text-center">
      <div className="text-2xl mb-1">{icons[agent.agent] || "🤖"}</div>
      <p className="text-xs text-gray-400 capitalize mb-2">
        {agent.agent.replace("_", " ")}
      </p>
      <div className="flex items-center justify-center gap-1">
        <div className={`w-1.5 h-1.5 rounded-full ${statusColors[agent.status] || "bg-gray-500"}`} />
        <span className="text-[10px] text-gray-500 capitalize">{agent.status}</span>
      </div>
    </div>
  );
}

function ActionButton({ label, icon, action }: { label: string; icon: string; action: string }) {
  const handleClick = async () => {
    try {
      await fetch(`${API_URL}/api/v1/stream/control`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
      });
    } catch {
      // handle error
    }
  };

  return (
    <button
      onClick={handleClick}
      className="px-4 py-2 bg-sf-border/50 hover:bg-sf-accent/20 border border-sf-border rounded-lg text-sm text-gray-300 hover:text-white transition-colors"
    >
      {icon} {label}
    </button>
  );
}

function PipelineButton({ label, genre }: { label: string; genre: string | undefined }) {
  const handleClick = async () => {
    try {
      if (genre === "batch") {
        await fetch(`${API_URL}/api/v1/pipeline/batch?count=5`, { method: "POST" });
      } else {
        const params = genre ? `?genre=${genre}` : "";
        await fetch(`${API_URL}/api/v1/pipeline/run${params}`, { method: "POST" });
      }
    } catch {
      // handle error
    }
  };

  return (
    <button
      onClick={handleClick}
      className="px-4 py-3 bg-sf-accent/10 hover:bg-sf-accent/20 border border-sf-accent/30 rounded-lg text-sm text-sf-accent-light hover:text-white transition-colors"
    >
      {label}
    </button>
  );
}

const defaultAgents = [
  { agent: "composer", status: "idle" },
  { agent: "producer", status: "idle" },
  { agent: "critic", status: "idle" },
  { agent: "scheduler", status: "idle" },
  { agent: "stream_master", status: "idle" },
  { agent: "analytics", status: "idle" },
  { agent: "visual", status: "idle" },
];
