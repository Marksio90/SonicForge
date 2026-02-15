'use client';

import { useEffect, useState, useCallback, useRef } from 'react';

interface LiveData {
  stream: {
    status: string;
    viewers: number;
    current_track: {
      title: string;
      genre: string;
      bpm: number;
      progress: number;
      duration: number;
    };
  };
  metrics: {
    tracks_generated_today: number;
    total_plays: number;
    queue_length: number;
    active_agents: number;
  };
  system: {
    cpu_usage: number;
    memory_usage: number;
    latency_ms: number;
  };
  agents: Array<{
    id: string;
    name: string;
    status: string;
    icon: string;
  }>;
  recent_events: Array<{
    type: string;
    message: string;
    timestamp: string;
  }>;
  genre_distribution: Record<string, number>;
}

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8001';

export default function LiveDashboard() {
  const [data, setData] = useState<LiveData | null>(null);
  const [connected, setConnected] = useState(false);
  const [reconnecting, setReconnecting] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(`${WS_URL}/ws/live`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
      setReconnecting(false);
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        
        if (message.type === 'initial_state' || message.type === 'full_state') {
          setData(message.data);
        } else if (message.type === 'live_update' && data) {
          // Merge incremental updates
          setData(prev => prev ? {
            ...prev,
            stream: {
              ...prev.stream,
              viewers: message.data.viewers,
              current_track: message.data.current_track,
            },
            system: message.data.system,
            agents: message.data.agents,
          } : prev);
        }
      } catch (e) {
        console.error('Parse error:', e);
      }
    };

    ws.onerror = () => {
      console.error('WebSocket error');
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setConnected(false);
      
      // Auto reconnect after 3 seconds
      if (!reconnecting) {
        setReconnecting(true);
        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, 3000);
      }
    };
  }, [reconnecting]);

  useEffect(() => {
    connect();
    
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  if (!data) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-sf-accent border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Connecting to live feed...</p>
        </div>
      </div>
    );
  }

  const trackProgress = (data.stream.current_track.progress / data.stream.current_track.duration) * 100;

  return (
    <div className="space-y-6">
      {/* Header with Connection Status */}
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Live Dashboard</h1>
          <p className="text-gray-500 text-sm mt-1">
            Real-time monitoring • Updates every second
          </p>
        </div>
        <div className="flex items-center gap-4">
          <ConnectionBadge connected={connected} reconnecting={reconnecting} />
          <LiveBadge status={data.stream.status} viewers={data.stream.viewers} />
        </div>
      </header>

      {/* Live Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard 
          label="Viewers" 
          value={data.stream.viewers} 
          icon="👥" 
          color="cyan"
          animate 
        />
        <MetricCard 
          label="Tracks Today" 
          value={data.metrics.tracks_generated_today} 
          icon="🎵" 
          color="accent"
        />
        <MetricCard 
          label="Total Plays" 
          value={data.metrics.total_plays.toLocaleString()} 
          icon="▶️" 
          color="green"
        />
        <MetricCard 
          label="Queue" 
          value={`${data.metrics.queue_length} tracks`} 
          icon="📋" 
          color={data.metrics.queue_length > 5 ? 'green' : 'yellow'}
        />
      </div>

      {/* Now Playing with Live Progress */}
      <div className="bg-sf-surface rounded-xl border border-sf-border p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white">Now Playing</h2>
          <span className="text-xs text-gray-500">
            {formatTime(data.stream.current_track.progress)} / {formatTime(data.stream.current_track.duration)}
          </span>
        </div>
        
        <div className="flex items-center gap-6">
          <div className="w-20 h-20 rounded-xl bg-gradient-to-br from-sf-accent/30 to-sf-cyan/30 flex items-center justify-center">
            <span className="text-4xl">🎵</span>
          </div>
          
          <div className="flex-1">
            <h3 className="text-xl text-white font-medium">{data.stream.current_track.title}</h3>
            <p className="text-gray-400 text-sm">
              {data.stream.current_track.genre} • {data.stream.current_track.bpm} BPM
            </p>
            
            {/* Progress Bar */}
            <div className="mt-3 h-2 bg-sf-border rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-sf-accent to-sf-cyan transition-all duration-1000"
                style={{ width: `${trackProgress}%` }}
              />
            </div>
          </div>
          
          <div className="text-right">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-sf-accent/20 rounded-full">
              <div className="w-2 h-2 bg-sf-accent rounded-full animate-pulse" />
              <span className="text-sm text-sf-accent-light">Live</span>
            </div>
          </div>
        </div>
      </div>

      {/* System Health & Agents */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* System Metrics */}
        <div className="bg-sf-surface rounded-xl border border-sf-border p-6">
          <h2 className="text-lg font-semibold text-white mb-4">System Health</h2>
          <div className="space-y-4">
            <SystemBar label="CPU Usage" value={data.system.cpu_usage} unit="%" color="cyan" />
            <SystemBar label="Memory" value={data.system.memory_usage} unit="%" color="accent" />
            <SystemBar label="Latency" value={data.system.latency_ms} unit="ms" max={100} color="green" />
          </div>
        </div>

        {/* Agent Grid */}
        <div className="bg-sf-surface rounded-xl border border-sf-border p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-white">AI Agents</h2>
            <span className="text-xs text-gray-500">
              {data.metrics.active_agents}/{data.agents.length} active
            </span>
          </div>
          <div className="grid grid-cols-4 md:grid-cols-7 gap-2">
            {data.agents.map((agent) => (
              <AgentPill key={agent.id} agent={agent} />
            ))}
          </div>
        </div>
      </div>

      {/* Recent Events */}
      <div className="bg-sf-surface rounded-xl border border-sf-border p-6">
        <h2 className="text-lg font-semibold text-white mb-4">Live Activity Feed</h2>
        <div className="space-y-2 max-h-48 overflow-y-auto">
          {data.recent_events.length > 0 ? (
            data.recent_events.slice().reverse().map((event, i) => (
              <div 
                key={i} 
                className={`flex items-center gap-3 py-2 px-3 rounded-lg ${i === 0 ? 'bg-sf-accent/10 border border-sf-accent/20' : 'hover:bg-sf-border/30'}`}
              >
                <EventIcon type={event.type} />
                <span className="text-sm text-gray-300 flex-1">{event.message}</span>
                <span className="text-xs text-gray-500">
                  {new Date(event.timestamp).toLocaleTimeString()}
                </span>
              </div>
            ))
          ) : (
            <p className="text-gray-500 text-sm text-center py-4">
              Waiting for events...
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

// Sub-components

function ConnectionBadge({ connected, reconnecting }: { connected: boolean; reconnecting: boolean }) {
  if (reconnecting) {
    return (
      <div className="flex items-center gap-2 px-3 py-1.5 bg-yellow-500/10 border border-yellow-500/30 rounded-full">
        <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse" />
        <span className="text-xs text-yellow-400">Reconnecting...</span>
      </div>
    );
  }
  
  return (
    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${
      connected 
        ? 'bg-green-500/10 border border-green-500/30' 
        : 'bg-red-500/10 border border-red-500/30'
    }`}>
      <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`} />
      <span className={`text-xs ${connected ? 'text-green-400' : 'text-red-400'}`}>
        {connected ? 'Connected' : 'Disconnected'}
      </span>
    </div>
  );
}

function LiveBadge({ status, viewers }: { status: string; viewers: number }) {
  const isLive = status === 'live';
  
  return (
    <div className={`flex items-center gap-3 px-4 py-2 rounded-full ${
      isLive ? 'bg-red-500/10 border border-red-500/30' : 'bg-gray-500/10 border border-gray-500/30'
    }`}>
      <div className="flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${isLive ? 'bg-red-500 animate-pulse' : 'bg-gray-500'}`} />
        <span className={`text-sm font-medium ${isLive ? 'text-red-400' : 'text-gray-400'}`}>
          {isLive ? 'LIVE' : 'OFFLINE'}
        </span>
      </div>
      {isLive && (
        <span className="text-sm text-gray-400">
          {viewers.toLocaleString()} viewers
        </span>
      )}
    </div>
  );
}

function MetricCard({ label, value, icon, color, animate }: { 
  label: string; 
  value: string | number; 
  icon: string; 
  color: string;
  animate?: boolean;
}) {
  const colorClasses: Record<string, string> = {
    cyan: 'text-sf-cyan border-sf-cyan/30',
    accent: 'text-sf-accent border-sf-accent/30',
    green: 'text-sf-green border-sf-green/30',
    yellow: 'text-sf-yellow border-sf-yellow/30',
    red: 'text-sf-red border-sf-red/30',
  };

  return (
    <div className={`bg-sf-surface rounded-xl border p-4 ${colorClasses[color]}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-gray-500 uppercase tracking-wider">{label}</span>
        <span className="text-lg">{icon}</span>
      </div>
      <p className={`text-2xl font-bold ${colorClasses[color].split(' ')[0]} ${animate ? 'transition-all duration-300' : ''}`}>
        {value}
      </p>
    </div>
  );
}

function SystemBar({ label, value, unit, max = 100, color }: {
  label: string;
  value: number;
  unit: string;
  max?: number;
  color: string;
}) {
  const percentage = Math.min((value / max) * 100, 100);
  const colorClasses: Record<string, string> = {
    cyan: 'from-sf-cyan to-sf-cyan/50',
    accent: 'from-sf-accent to-sf-accent/50',
    green: 'from-sf-green to-sf-green/50',
  };

  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-gray-400">{label}</span>
        <span className="text-white font-medium">{value.toFixed(1)}{unit}</span>
      </div>
      <div className="h-2 bg-sf-border rounded-full overflow-hidden">
        <div 
          className={`h-full bg-gradient-to-r ${colorClasses[color]} transition-all duration-500`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

function AgentPill({ agent }: { agent: { id: string; name: string; status: string; icon: string } }) {
  const statusColors: Record<string, string> = {
    idle: 'bg-sf-green',
    working: 'bg-sf-yellow animate-pulse',
    error: 'bg-sf-red',
    stopped: 'bg-gray-600',
  };

  return (
    <div className="flex flex-col items-center p-2 rounded-lg bg-sf-bg border border-sf-border hover:border-sf-accent/50 transition-colors">
      <span className="text-xl mb-1">{agent.icon}</span>
      <div className={`w-2 h-2 rounded-full ${statusColors[agent.status] || 'bg-gray-500'}`} />
    </div>
  );
}

function EventIcon({ type }: { type: string }) {
  const icons: Record<string, string> = {
    track_generated: '🎵',
    agent_status: '🤖',
    viewer_milestone: '🎉',
    error: '⚠️',
    default: '📌',
  };

  return <span className="text-lg">{icons[type] || icons.default}</span>;
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}
