"""Tests for SonicForge agent system."""
import pytest

from app.agents.composer import ComposerAgent
from app.agents.critic import CriticAgent
from app.agents.scheduler import SchedulerAgent
from app.agents.visual import VisualAgent
from app.core.config import Genre


@pytest.mark.asyncio
async def test_composer_creates_concept():
    composer = ComposerAgent()
    concept = await composer.execute({
        "type": "create_concept",
        "genre": "drum_and_bass",
    })

    assert concept["genre"] == "drum_and_bass"
    assert 170 <= concept["bpm"] <= 180
    assert concept["key"] in [
        "C minor", "C# minor", "D minor", "D# minor", "E minor", "F minor",
        "F# minor", "G minor", "G# minor", "A minor", "A# minor", "B minor",
        "C major", "D major", "F major", "G major", "A major",
    ]
    assert "prompt" in concept
    assert len(concept["instruments"]) > 0


@pytest.mark.asyncio
async def test_composer_all_genres():
    composer = ComposerAgent()
    for genre in Genre:
        concept = await composer.execute({
            "type": "create_concept",
            "genre": genre.value,
        })
        assert concept["genre"] == genre.value
        assert concept["bpm"] > 0
        assert concept["prompt"]


@pytest.mark.asyncio
async def test_critic_evaluates_track():
    critic = CriticAgent()
    result = await critic.execute({
        "type": "evaluate",
        "track_id": "test-track-001",
        "genre": "drum_and_bass",
        "audio_data": None,
    })

    assert "overall_score" in result
    assert "approved" in result
    assert "scores" in result
    assert "feedback" in result
    assert isinstance(result["overall_score"], float)
    assert 0 <= result["overall_score"] <= 10


@pytest.mark.asyncio
async def test_critic_batch_evaluation():
    critic = CriticAgent()
    variants = [
        {"track_id": f"test-{i}", "genre": "house_deep"}
        for i in range(3)
    ]
    result = await critic.execute({
        "type": "evaluate_batch",
        "variants": variants,
    })

    assert result["total_count"] == 3
    assert result["best"] is not None
    assert "overall_score" in result["best"]


@pytest.mark.asyncio
async def test_visual_generates_config():
    visual = VisualAgent()
    result = await visual.execute({
        "type": "generate_visual",
        "track_id": "test-track-001",
        "genre": "drum_and_bass",
        "bpm": 174,
        "title": "Test Track",
    })

    assert result["visual_theme"] == "cyberpunk_neon_fractals"
    assert "shader_config" in result
    assert "overlay_config" in result
    assert result["shader_config"]["bpm_sync"] == 174


@pytest.mark.asyncio
async def test_visual_overlay():
    visual = VisualAgent()
    result = await visual.execute({
        "type": "generate_overlay",
        "track_id": "test-001",
        "title": "Test DnB Track",
        "bpm": 174,
        "key": "D minor",
        "genre": "drum_and_bass",
        "score": 9.2,
    })

    assert "elements" in result
    assert len(result["elements"]) >= 2


@pytest.mark.asyncio
async def test_composer_trend_analysis():
    composer = ComposerAgent()
    result = await composer.execute({
        "type": "analyze_trends",
        "genre": "techno_melodic",
    })

    assert "trending_bpm" in result
    assert "trending_keys" in result


@pytest.mark.asyncio
async def test_composer_weighted_genre_select():
    """Test that weighted genre selection produces valid genres."""
    composer = ComposerAgent()
    for _ in range(20):
        genre = composer._weighted_genre_select()
        assert genre in [g.value for g in Genre]


@pytest.mark.asyncio
async def test_composer_energy_key_preferences():
    """Test that energy-aware key selection works for all energy levels."""
    composer = ComposerAgent()
    for energy in range(1, 6):
        concept = await composer.execute({
            "type": "create_concept",
            "genre": "drum_and_bass",
            "energy": energy,
        })
        assert concept["energy"] == energy
        assert concept["key"]  # key should always be set


@pytest.mark.asyncio
async def test_critic_parallel_evaluation():
    """Test that batch evaluation correctly ranks variants."""
    critic = CriticAgent()
    variants = [
        {"track_id": f"parallel-test-{i}", "genre": "trance_uplifting"}
        for i in range(5)
    ]
    result = await critic.execute({
        "type": "evaluate_batch",
        "variants": variants,
    })

    assert result["total_count"] == 5
    # Scores should be sorted descending
    evaluations = result["evaluations"]
    for i in range(len(evaluations) - 1):
        assert evaluations[i]["overall_score"] >= evaluations[i + 1]["overall_score"]
