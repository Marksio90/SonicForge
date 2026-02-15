# 🛠️ SonicForge Optimization — Implementation Guide

## Szczegółowy przewodnik implementacji wszystkich 115+ ulepszeń

---

## 🎯 Jak używać tego przewodnika

Każde ulepszenie zawiera:
- ✅ **Status:** Do zrobienia / W trakcie / Ukończone
- 🎯 **Priorytet:** Krytyczny / Wysoki / Średni / Niski
- ⏱️ **Czas:** Szacowany czas implementacji
- 📦 **Zależności:** Co musi być najpierw ukończone
- 🔧 **Kod:** Gotowe przykłady do implementacji
- ✔️ **Weryfikacja:** Jak sprawdzić, czy działa

---

## KATEGORIA 1: PERFORMANCE & SCALABILITY

### 1.1 Advanced Connection Pooling ⭐⭐⭐⭐⭐

**Status:** ❌ Do zrobienia  
**Priorytet:** 🔴 Krytyczny  
**Czas:** 4 godziny  
**Zależności:** Brak

#### Krok 1: Utwórz Connection Pool Manager

```bash
# Utwórz nowy plik
touch backend/app/core/connection_pool.py
```

Wklej kod z Master Plan (sekcja 1.1).

#### Krok 2: Zaktualizuj main.py

```python
# backend/app/main.py
from .core.connection_pool import pool_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize connection pools
    await pool_manager.initialize()
    
    yield
    
    # Close pools
    await pool_manager.close()
```

#### Krok 3: Użyj w agentach

```python
# backend/app/agents/producer.py
from app.core.connection_pool import pool_manager

async def call_api(self, url: str):
    client = await pool_manager.get_http_client()
    response = await client.get(url)
    return response
```

#### Weryfikacja

```bash
# Test connection pooling
python -m pytest tests/test_connection_pool.py -v

# Check connections in use
watch -n 1 'psql -c "SELECT count(*) FROM pg_stat_activity"'
```

**Expected Result:** Connections should be reused, not created for each request.

---

### 1.2 Multi-Layer Caching ⭐⭐⭐⭐⭐

**Status:** ❌ Do zrobienia  
**Priorytet:** 🔴 Krytyczny  
**Czas:** 6 godzin  
**Zależności:** 1.1 (Connection Pool)

#### Krok 1: Utwórz Cache Layer

```bash
touch backend/app/core/cache.py
```

Wklej kod z Master Plan (sekcja 1.2).

#### Krok 2: Initialize w main.py

```python
# backend/app/main.py
from .core.cache import cache

@asynccontextmanager
async def lifespan(app: FastAPI):
    await cache.initialize(settings.redis_url)
    yield
```

#### Krok 3: Użyj dekoratora @cached

```python
# backend/app/agents/composer.py
from app.core.cache import cached

@cached(prefix="trend_analysis", ttl=3600)
async def analyze_trends(self, genre: str | None = None) -> dict:
    # This expensive operation is now cached for 1 hour
    client = self._get_openai_client()
    ...
```

#### Krok 4: Cache invalidation

```python
# Invalidate cache when needed
await cache.invalidate("trend_analysis")
```

#### Weryfikacja

```bash
# Monitor cache hit rate
redis-cli
> INFO stats
> KEYS trend_analysis:*

# Check memory cache size
# Add logging in cache.py to track hits/misses
```

**Expected Result:** 
- First call: slow (cache miss)
- Second call: <10ms (cache hit)
- Redis should show keys with TTL

---

### 1.3 Horizontal Scaling with Kubernetes ⭐⭐⭐⭐

**Status:** ❌ Do zrobienia  
**Priorytet:** 🟡 Wysoki  
**Czas:** 8 godzin  
**Zależności:** Docker, Kubernetes cluster

#### Krok 1: Przygotuj Kubernetes manifesty

```bash
mkdir -p kubernetes
touch kubernetes/deployment.yaml
touch kubernetes/service.yaml
touch kubernetes/hpa.yaml
```

Wklej kod z Master Plan (sekcja 1.3).

#### Krok 2: Zbuduj Docker image

```dockerfile
# backend/Dockerfile (optimized)
FROM python:3.12-slim as builder

WORKDIR /app

# Install dependencies
COPY backend/pyproject.toml .
RUN pip install --user --no-cache-dir -e .

# Runtime stage
FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY backend/app ./app

ENV PATH=/root/.local/bin:$PATH

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Krok 3: Deploy do Kubernetes

```bash
# Build and push image
docker build -t sonicforge/api:3.0 -f backend/Dockerfile .
docker push sonicforge/api:3.0

# Apply Kubernetes manifests
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/hpa.yaml

# Check status
kubectl get pods -l app=sonicforge
kubectl get hpa
```

#### Krok 4: Test auto-scaling

```bash
# Generate load
hey -z 60s -c 100 http://your-service-url/api/v1/dashboard/overview

# Watch pods scale up
watch kubectl get pods
```

#### Weryfikacja

```bash
# Check if HPA is working
kubectl describe hpa sonicforge-api-hpa

# Should see:
# - Current CPU: 75%
# - Desired replicas: 8
# - Pods scaling up
```

**Expected Result:** 
- Under load: scales from 3 to 10+ pods
- After load: scales down to 3 pods (after stabilization window)

---

### 1.4 Database Query Optimization ⭐⭐⭐⭐⭐

**Status:** ❌ Do zrobienia  
**Priorytet:** 🔴 Krytyczny  
**Czas:** 6 godzin  
**Zależności:** Brak

#### Krok 1: Dodaj indexy do modeli

```python
# backend/app/models/track.py
# Update Track model with indexes from Master Plan (sekcja 1.4)
```

#### Krok 2: Utwórz migration

```bash
cd backend
alembic revision --autogenerate -m "Add performance indexes"
alembic upgrade head
```

#### Krok 3: Zoptymalizuj queries

```python
# backend/app/services/track_service.py
from sqlalchemy.orm import load_only

async def get_tracks_for_dashboard():
    """Optimized query - only load needed columns."""
    result = await session.execute(
        select(Track)
        .options(load_only(Track.id, Track.genre, Track.score))
        .where(Track.approved == True)
        .order_by(Track.created_at.desc())
        .limit(50)
    )
    return result.scalars().all()
```

#### Krok 4: Analyze query plans

```python
# Add EXPLAIN ANALYZE to slow queries
from sqlalchemy import text

result = await session.execute(
    text("EXPLAIN ANALYZE SELECT * FROM tracks WHERE genre = :genre"),
    {"genre": "drum_and_bass"}
)
print(result.fetchall())
```

#### Weryfikacja

```sql
-- Check if indexes are being used
EXPLAIN ANALYZE
SELECT * FROM tracks 
WHERE genre = 'drum_and_bass' AND approved = true
ORDER BY score DESC
LIMIT 10;

-- Should show "Index Scan using idx_genre_score"
```

**Expected Result:**
- Query time: 500ms → 10-50ms
- Index scans instead of sequential scans
- No "Seq Scan" in EXPLAIN output

---

### 1.5 Celery Task Priorities & Rate Limiting ⭐⭐⭐⭐

**Status:** ❌ Do zrobienia  
**Priorytet:** 🟡 Wysoki  
**Czas:** 4 godziny  
**Zależności:** Redis

#### Krok 1: Zaktualizuj Celery config

```python
# backend/app/core/celery_app.py
# Replace with enhanced config from Master Plan (sekcja 1.5)
```

#### Krok 2: Dodaj priority do tasków

```python
# backend/app/services/tasks.py
from app.core.celery_app import celery_app, RateLimitedTask

@celery_app.task(base=RateLimitedTask, queue="high", priority=7)
def generate_track_urgent(concept_id: str):
    """High-priority track generation."""
    pass

@celery_app.task(base=RateLimitedTask, queue="low", priority=2)
def cleanup_old_files():
    """Low-priority cleanup."""
    pass
```

#### Krok 3: Uruchom workery z priorytetami

```bash
# Start workers for different queues
celery -A app.core.celery_app worker -Q critical,high -c 4
celery -A app.core.celery_app worker -Q default,low,batch -c 2
```

#### Weryfikacja

```python
# Send tasks with different priorities
generate_track_urgent.apply_async(args=["concept1"], priority=9)
cleanup_old_files.apply_async(priority=1)

# High priority task should execute first
```

**Expected Result:**
- High-priority tasks execute immediately
- Low-priority tasks wait
- Rate limiting prevents overload

---

## KATEGORIA 2: AI/ML OPTIMIZATION

### 2.1 Model Quantization ⭐⭐⭐⭐⭐

**Status:** ❌ Do zrobienia  
**Priorytet:** 🔴 Krytyczny  
**Czas:** 8 godzin  
**Zależności:** GPU, PyTorch 2.0+

#### Krok 1: Zainstaluj dependencies

```bash
pip install torch>=2.1.0 torchvision torchaudio
pip install transformers accelerate bitsandbytes
```

#### Krok 2: Utwórz optimized engine

```bash
touch backend/app/services/musicgen_optimized.py
```

Wklej kod z Master Plan (sekcja 2.1).

#### Krok 3: Zaktualizuj Producer Agent

```python
# backend/app/agents/producer.py
from app.services.musicgen_optimized import OptimizedMusicGenEngine

class ProducerAgent(BaseAgent):
    def __init__(self):
        super().__init__("producer")
        self._musicgen = OptimizedMusicGenEngine()
        self._musicgen.load_model()  # Load optimized model at startup
```

#### Krok 4: Test performance

```python
import time

# Before optimization
start = time.time()
audio = await musicgen.generate(concept)
print(f"Time: {time.time() - start:.2f}s")  # ~150s

# After optimization
start = time.time()
audio = await optimized_musicgen.generate(concept)
print(f"Time: {time.time() - start:.2f}s")  # ~30-50s
```

#### Weryfikacja

```bash
# Monitor GPU memory usage
watch -n 1 nvidia-smi

# Before: ~20GB VRAM
# After: ~10GB VRAM (50% reduction)
```

**Expected Result:**
- Generation time: 150s → 30-50s (3-5x faster)
- Memory usage: 20GB → 10GB (50% reduction)
- Quality loss: <5%

---

### 2.2 Batch Processing ⭐⭐⭐⭐

**Status:** ❌ Do zrobienia  
**Priorytet:** 🟡 Wysoki  
**Czas:** 6 godzin  
**Zależności:** 2.1 (Optimized Model)

#### Krok 1: Dodaj batch endpoint do MusicGen service

```python
# musicgen_service/app/main.py
@app.post("/generate_batch")
async def generate_batch(request: BatchRequest):
    """Generate multiple tracks in single batch."""
    prompts = request.prompts
    
    # Process all prompts in single forward pass
    audio_batch = await model.generate_batch(prompts)
    
    return {"audio_batch": audio_batch}
```

#### Krok 2: Update Producer Agent

Wklej kod batch processing z Master Plan (sekcja 2.2).

#### Krok 3: Test batch vs sequential

```python
# Sequential (old way)
start = time.time()
for i in range(5):
    await producer.generate_track(concept)
print(f"Sequential: {time.time() - start:.2f}s")  # ~250s

# Batch (new way)
start = time.time()
await producer.generate_track_batch([concept] * 5)
print(f"Batch: {time.time() - start:.2f}s")  # ~80-100s
```

#### Weryfikacja

**Expected Result:**
- 5 tracks sequential: ~250s
- 5 tracks batched: ~80-100s (2-3x faster)
- Better GPU utilization (90%+ vs 50%)

---

### 2.3 Model Ensemble ⭐⭐⭐

**Status:** ❌ Do zrobienia  
**Priorytet:** 🟢 Średni  
**Czas:** 10 godzin  
**Zależności:** Multiple model APIs

#### Implementation steps w Master Plan (sekcja 2.3)

---

## KATEGORIA 3: AUDIO PROCESSING

### 3.1 Pro Mastering Engine ⭐⭐⭐⭐

**Status:** ❌ Do zrobienia  
**Priorytet:** 🟡 Wysoki  
**Czas:** 12 godzin  
**Zależności:** librosa, scipy

#### Krok 1: Rozbuduj mastering chain

```bash
touch backend/app/services/mastering_pro.py
```

Wklej kod z Master Plan (sekcja 3.1).

#### Krok 2: Implementuj wszystkie procesory

```python
def de_esser(self, audio: np.ndarray) -> np.ndarray:
    """Remove harsh high frequencies."""
    from scipy.signal import butter, filtfilt
    
    # High-shelf filter at 6 kHz with -3dB
    sos = butter(4, 6000/(44100/2), btype='high', output='sos')
    filtered = filtfilt(sos, audio, axis=0)
    
    # Blend original with filtered (50% reduction)
    return audio * 0.7 + filtered * 0.3
```

#### Weryfikacja

```python
# Compare before/after
import soundfile as sf

# Original
sf.write("before.wav", audio_before, 44100)

# Mastered
mastered = await mastering_engine.master(audio_before)
sf.write("after.wav", mastered, 44100)

# Listen and check:
# - Louder overall (LUFS normalized)
# - Clearer highs (de-essing)
# - Balanced frequencies (multiband)
# - No clipping (limiter)
```

---

## KATEGORIA 4: MONITORING & OBSERVABILITY

### 5.1 Distributed Tracing ⭐⭐⭐⭐

**Status:** ❌ Do zrobienia  
**Priorytet:** 🟡 Wysoki  
**Czas:** 6 godzin  
**Zależności:** Jaeger

#### Krok 1: Deploy Jaeger

```bash
docker run -d --name jaeger \
  -p 5775:5775/udp \
  -p 6831:6831/udp \
  -p 6832:6832/udp \
  -p 5778:5778 \
  -p 16686:16686 \
  -p 14268:14268 \
  -p 9411:9411 \
  jaegertracing/all-in-one:latest
```

#### Krok 2: Setup OpenTelemetry

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-jaeger
```

```python
# backend/app/core/tracing.py - kod z Master Plan (sekcja 5.1)
```

#### Krok 3: Instrument code

```python
# backend/app/services/orchestrator.py
from app.core.tracing import tracer

async def run_full_pipeline(self, genre: str):
    with tracer.start_as_current_span("pipeline"):
        with tracer.start_as_current_span("compose"):
            concept = await self.composer.run({"genre": genre})
        
        with tracer.start_as_current_span("produce"):
            tracks = await self.producer.run({"concept": concept})
        
        return tracks
```

#### Weryfikacja

```bash
# Generate some traffic
curl -X POST http://localhost:8000/api/v1/pipeline/run?genre=drum_and_bass

# Open Jaeger UI
open http://localhost:16686

# Should see:
# - Service: sonicforge-api
# - Trace with spans: pipeline → compose → produce
# - Timing for each step
```

---

## QUICK START: Top 20 Priority Implementations

Jeśli chcesz zacząć od najważniejszych optymalizacji:

### Week 1: Foundation (MUST HAVE)
1. ✅ 1.1 Connection Pooling (4h)
2. ✅ 1.2 Multi-Layer Caching (6h)
3. ✅ 1.4 Database Indexing (6h)
4. ✅ 2.1 Model Quantization (8h)
5. ✅ 1.5 Task Priorities (4h)

### Week 2: Scaling (HIGH PRIORITY)
6. ✅ 1.3 Kubernetes Deployment (8h)
7. ✅ 1.6 Redis Sentinel (4h)
8. ✅ 2.2 Batch Processing (6h)
9. ✅ 1.7 CDN Integration (6h)
10. ✅ 5.1 Distributed Tracing (6h)

### Week 3: Advanced (NICE TO HAVE)
11. ✅ 2.3 Model Ensemble (10h)
12. ✅ 3.1 Pro Mastering (12h)
13. ✅ 4.1 Multi-Platform Streaming (8h)
14. ✅ 6.1 Real-time Dashboard (10h)
15. ✅ 7.1 Secrets Management (4h)

### Week 4: Testing & Polish
16. ✅ 8.1 Unit Tests (8h)
17. ✅ 8.3 Load Testing (6h)
18. ✅ 9.1 Database Partitioning (6h)
19. ✅ 10.1 Cost Optimization (8h)
20. ✅ 12.3 CI/CD Pipeline (8h)

**Total:** ~132 hours (~3-4 weeks with 2 engineers)

---

## Tracking Progress

Użyj tego checklisty do śledzenia postępu:

```markdown
## Performance & Scalability
- [ ] 1.1 Connection Pooling
- [ ] 1.2 Multi-Layer Caching
- [ ] 1.3 Horizontal Scaling
- [ ] 1.4 Database Optimization
- [ ] 1.5 Task Priorities
- [ ] 1.6 Redis HA
- [ ] 1.7 CDN Integration
- [ ] 1.8 Read Replicas
- [ ] 1.9-1.15 Misc Optimizations

## AI/ML Optimization
- [ ] 2.1 Model Quantization
- [ ] 2.2 Batch Processing
- [ ] 2.3 Model Ensemble
- [ ] 2.4 Prompt Engineering
- [ ] 2.5 Embedding Cache
- [ ] 2.6-2.12 Misc AI Optimizations

## Audio Processing
- [ ] 3.1 Pro Mastering
- [ ] 3.2 Real-time Analysis
- [ ] 3.3 Fingerprinting
- [ ] 3.4 Adaptive Bitrate
- [ ] 3.5-3.10 Misc Audio Features

... (continue for all categories)
```

---

## 🆘 Troubleshooting

### Problem: Connection pool exhausted
**Solution:** Increase `max_size` in pool config or add more replicas

### Problem: Cache not working
**Solution:** Check Redis connection, verify keys with `redis-cli KEYS *`

### Problem: Kubernetes pods not scaling
**Solution:** Check HPA metrics: `kubectl describe hpa`, verify metrics-server

### Problem: Model too slow despite quantization
**Solution:** 
- Enable torch.compile
- Use GPU instead of CPU
- Check CUDA drivers

### Problem: High memory usage
**Solution:**
- Enable model quantization
- Use batch processing
- Clear cache periodically

---

## 📚 Resources

- [FastAPI Best Practices](https://fastapi.tiangolo.com/advanced/)
- [Kubernetes Patterns](https://kubernetes.io/docs/concepts/)
- [PyTorch Performance](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Database Optimization](https://wiki.postgresql.org/wiki/Performance_Optimization)
- [Redis Best Practices](https://redis.io/docs/management/optimization/)

---

**Last Updated:** 2025-08-XX  
**Maintainer:** Engineering Team  
**Status:** 🚀 Ready to Implement

