# SonicForge

**AI-Powered 24/7 Music Radio Platform**

Autonomiczne radio muzyczne generujące, kuratorujące i streamujące muzykę elektroniczną najwyższej jakości 24/7 na YouTube Live. Nie playlist — żywy organizm prowadzony przez inteligentne agenty AI.

## Architecture

```
┌──────────────────────────────────────────────────┐
│              ORCHESTRATOR (FastAPI)               │
│         Centralny mózg systemu                   │
├──────┬──────┬──────┬──────┬──────┬──────┬────────┤
│Composer│Producer│Critic│Scheduler│Stream│Analytics│Visual│
│ (LLM) │(Music │ (QA) │(Timing) │Master│ (Data) │(Art) │
│       │GenAPI)│      │         │(RTMP)│        │      │
└──┬────┴──┬───┴──┬───┴──┬──────┴──┬───┴──┬─────┴──┬───┘
   │       │      │      │         │      │        │
   └───────┴──────┴──────┴─────────┴──────┴────────┘
              Redis + PostgreSQL + MinIO
```

## Multi-Agent System

| Agent | Role | Key Features |
|-------|------|--------------|
| **Composer** | Music concept creation | Trend analysis, prompt engineering, structure design |
| **Producer** | Audio generation | Suno/Udio/ElevenLabs APIs, multi-variant generation |
| **Critic** | Quality gate | Spectral analysis, 1-10 scoring, artifact detection |
| **Scheduler** | 24/7 playlist | Time-of-day energy, genre flow, listener requests |
| **StreamMaster** | Live streaming | FFmpeg→RTMP, auto-recovery, health monitoring |
| **Analytics** | Intelligence | YouTube API, viewer tracking, A/B testing |
| **Visual** | Visualizations | Audio-reactive shaders, overlays, thumbnails |

## Supported Genres

Drum & Bass · Liquid DnB · Melodic Dubstep · Deep House · Progressive House · Uplifting Trance · Psytrance · Melodic Techno · Breakbeat · Ambient · Downtempo

## Quick Start

### Prerequisites

- Docker & Docker Compose
- API keys (see `.env.example`)

### Setup

```bash
# Clone and configure
cp .env.example .env
# Edit .env with your API keys (Suno, Claude/OpenAI, YouTube)

# Start all services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f api
```

### Access Points

| Service | URL |
|---------|-----|
| **Dashboard** | http://localhost:3000 |
| **API (Swagger)** | http://localhost:8000/docs |
| **Grafana** | http://localhost:3001 |
| **MinIO Console** | http://localhost:9001 |
| **Prometheus** | http://localhost:9090 |

### API Quick Examples

```bash
# Generate a track (async)
curl -X POST "http://localhost:8000/api/v1/pipeline/run?genre=drum_and_bass"

# Create a concept
curl -X POST "http://localhost:8000/api/v1/tracks/concept" \
  -H "Content-Type: application/json" \
  -d '{"genre": "trance_uplifting", "energy": 4}'

# Check stream status
curl http://localhost:8000/api/v1/stream/status

# Get queue
curl http://localhost:8000/api/v1/stream/queue

# Start the stream
curl -X POST "http://localhost:8000/api/v1/stream/control" \
  -H "Content-Type: application/json" \
  -d '{"action": "start"}'

# List genres
curl http://localhost:8000/api/v1/genres
```

## Tech Stack

**Backend:** Python 3.12, FastAPI, Celery, Redis, PostgreSQL, MinIO (S3)
**AI Music:** Suno API, Udio API, ElevenLabs Music
**LLM:** Claude API / GPT-4 (prompt crafting, chat bot)
**Audio:** Librosa, FFmpeg (analysis, mastering, streaming)
**Streaming:** FFmpeg → RTMP → YouTube Live, Gyre.pro backup
**Dashboard:** Next.js 15, Tailwind CSS, Recharts, WebSocket
**Monitoring:** Grafana, Prometheus, Sentry, Telegram Bot
**DevOps:** Docker Compose, GitHub Actions CI/CD

## Production Pipeline

1. **Trend Analysis** — Composer analyzes Beatport/Spotify charts
2. **Prompt Crafting** — Ultra-detailed prompt with genre, BPM, key, structure
3. **Multi-Generation** — 3-5 variants via Suno/Udio/ElevenLabs in parallel
4. **Quality Gate** — Critic scores 1-10, only ≥8.5 approved (top ~15-20%)
5. **Post-Production** — Mastering (LUFS normalization, EQ, limiting)
6. **Visual Generation** — Audio-reactive shaders per genre
7. **Schedule & Broadcast** — Time-aware scheduling → FFmpeg → YouTube Live

## Quality Philosophy

Every generated track passes through the Critic Agent:
- Spectral analysis (frequency balance)
- Structure evaluation (intro/build/drop/breakdown)
- AI artifact detection
- Genre conformity check
- Dynamic range analysis

**Only tracks scoring ≥ 8.5/10 make it to air.** Rejected tracks generate feedback for the Composer to improve subsequent generations.

## Project Structure

```
SonicForge/
├── backend/                  # Python FastAPI backend
│   ├── app/
│   │   ├── agents/          # 7 AI agents
│   │   ├── api/             # REST + WebSocket routes
│   │   ├── core/            # Config, DB, Redis, S3
│   │   ├── models/          # SQLAlchemy models
│   │   ├── schemas/         # Pydantic schemas
│   │   └── services/        # Orchestrator, Celery tasks
│   ├── alembic/             # Database migrations
│   └── tests/               # Test suite
├── dashboard/               # Next.js 15 control panel
│   └── src/
│       ├── app/             # Pages
│       ├── components/      # UI components
│       └── lib/             # API client
├── streaming/               # FFmpeg streaming engine
│   └── scripts/             # Stream loop, watchdog
├── monitoring/              # Grafana + Prometheus configs
├── docker-compose.yml       # Full orchestration
└── .github/workflows/       # CI/CD pipeline
```

## License

MIT
