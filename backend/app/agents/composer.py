import random

import httpx
import structlog

from ..core.config import GENRE_PROFILES, Genre, get_settings
from .base import BaseAgent

settings = get_settings()
logger = structlog.get_logger(__name__)

# Musical keys commonly used in electronic music
MUSICAL_KEYS = [
    "C minor", "C# minor", "D minor", "D# minor", "E minor", "F minor",
    "F# minor", "G minor", "G# minor", "A minor", "A# minor", "B minor",
    "C major", "D major", "F major", "G major", "A major",
]

STRUCTURE_TEMPLATES = {
    Genre.DRUM_AND_BASS: "8-bar intro → 16-bar build with rising tension → heavy drop with rolling breaks → 8-bar breakdown → second drop with variation → 8-bar outro",
    Genre.LIQUID_DNB: "16-bar atmospheric intro → smooth build with lush pads → flowing drop with liquid bassline → melodic breakdown with piano/vocal → second drop → gentle outro",
    Genre.DUBSTEP_MELODIC: "8-bar cinematic intro → melodic build with emotional chords → heavy melodic drop → 16-bar mid-section with arpeggios → second drop with variation → outro",
    Genre.HOUSE_DEEP: "16-bar minimal intro with kick → gradual layering of elements → deep groove section → breakdown with pads → main section → smooth outro",
    Genre.HOUSE_PROGRESSIVE: "16-bar intro → slow build adding layers every 8 bars → first climax → breakdown → bigger build → main drop → extended outro",
    Genre.TRANCE_UPLIFTING: "16-bar intro → epic build with supersaw leads → massive uplifting drop → emotional breakdown with piano → euphoric second drop → outro",
    Genre.TRANCE_PSY: "8-bar intro → psychedelic build with acid lines → driving drop with complex rhythms → trippy breakdown → intense second drop → outro",
    Genre.TECHNO_MELODIC: "16-bar hypnotic intro → slow evolution of melodic elements → peak section → minimal breakdown → driving main section → gradual outro",
    Genre.BREAKBEAT: "8-bar intro with chopped breaks → build with funky elements → energetic drop → breakdown → second drop with variation → outro",
    Genre.AMBIENT: "long evolving pad intro → textural layers building slowly → main atmospheric section → evolving soundscape → gentle resolution",
    Genre.DOWNTEMPO: "chill intro with lo-fi elements → smooth groove builds → main section with deep bass → breakdown → final section → fade out",
}

INSTRUMENTATION = {
    Genre.DRUM_AND_BASS: ["rolling breaks", "reese bass", "sub bass", "amen break", "vocal chops", "atmospheric pads"],
    Genre.LIQUID_DNB: ["lush pads", "liquid bass", "piano", "female vocals", "strings", "smooth drums"],
    Genre.DUBSTEP_MELODIC: ["heavy bass wobble", "emotional chords", "orchestral elements", "vocal chops", "synth leads"],
    Genre.HOUSE_DEEP: ["deep kick", "warm bass", "subtle pads", "organic percussion", "filtered vocals", "rhodes"],
    Genre.HOUSE_PROGRESSIVE: ["layered synths", "driving bass", "progressive arpeggios", "white noise risers", "subtle vocal"],
    Genre.TRANCE_UPLIFTING: ["supersaw leads", "euphoric pads", "pluck synths", "piano", "epic strings", "female vocal"],
    Genre.TRANCE_PSY: ["acid bass", "psychedelic leads", "complex percussion", "atmospheric fx", "303 acid line"],
    Genre.TECHNO_MELODIC: ["minimal kick", "melodic synth loop", "atmospheric textures", "subtle percussion", "deep bass"],
    Genre.BREAKBEAT: ["chopped breaks", "funky bass", "stab synths", "vocal samples", "scratches"],
    Genre.AMBIENT: ["evolving pads", "granular textures", "field recordings", "reverb trails", "gentle drones"],
    Genre.DOWNTEMPO: ["lo-fi drums", "warm bass", "mellow keys", "vinyl crackle", "soft synths"],
}


class ComposerAgent(BaseAgent):
    """Creates music concepts, analyzes trends, and crafts generation prompts."""

    def __init__(self):
        super().__init__("composer")

    async def execute(self, task: dict) -> dict:
        task_type = task.get("type", "create_concept")

        if task_type == "create_concept":
            return await self.create_concept(
                genre=task.get("genre"),
                energy=task.get("energy"),
                mood=task.get("mood"),
            )
        elif task_type == "analyze_trends":
            return await self.analyze_trends(genre=task.get("genre"))
        elif task_type == "craft_prompt":
            return await self.craft_prompt(concept=task.get("concept"))
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def create_concept(
        self,
        genre: str | None = None,
        energy: int | None = None,
        mood: str | None = None,
    ) -> dict:
        """Create a complete music concept for track generation."""
        if genre is None:
            genre = random.choice(list(Genre)).value

        genre_enum = Genre(genre)
        profile = GENRE_PROFILES[genre_enum]
        bpm = random.randint(*profile["bpm_range"])
        key = random.choice(MUSICAL_KEYS)
        energy = energy or {"low": 2, "medium": 3, "high": 4}.get(profile["energy"], 3)

        instruments = random.sample(
            INSTRUMENTATION.get(genre_enum, ["synth", "bass", "drums"]),
            k=min(4, len(INSTRUMENTATION.get(genre_enum, []))),
        )

        structure = STRUCTURE_TEMPLATES.get(genre_enum, "intro → build → drop → breakdown → outro")

        concept = {
            "genre": genre,
            "subgenre": genre_enum.value,
            "bpm": bpm,
            "key": key,
            "energy": energy,
            "mood": mood or self._generate_mood(genre_enum, energy),
            "instruments": instruments,
            "structure": structure,
            "visual_theme": profile["visual_theme"],
        }

        concept["prompt"] = self._build_generation_prompt(concept)

        self.logger.info("concept_created", genre=genre, bpm=bpm, key=key)
        return concept

    async def analyze_trends(self, genre: str | None = None) -> dict:
        """Analyze current music trends (placeholder for Beatport/Spotify API integration)."""
        self.logger.info("analyzing_trends", genre=genre)
        return {
            "trending_bpm": random.randint(120, 180),
            "trending_keys": random.sample(MUSICAL_KEYS, 3),
            "trending_elements": ["atmospheric pads", "vocal chops", "rolling bass"],
            "source": "trend_analysis",
            "genre": genre,
        }

    async def craft_prompt(self, concept: dict) -> dict:
        """Use LLM to craft an ultra-detailed generation prompt."""
        if settings.anthropic_api_key:
            return await self._llm_craft_prompt(concept)
        return {"prompt": self._build_generation_prompt(concept)}

    async def _llm_craft_prompt(self, concept: dict) -> dict:
        """Use Claude/GPT to create an expert-level music generation prompt."""
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

        system_prompt = (
            "You are an expert electronic music producer. Create ultra-detailed prompts for "
            "AI music generation that capture the essence of professional productions. "
            "Include specific production techniques, sound design details, and arrangement ideas."
        )

        user_prompt = (
            f"Create a detailed music generation prompt for:\n"
            f"Genre: {concept['genre']}\n"
            f"BPM: {concept['bpm']}\n"
            f"Key: {concept['key']}\n"
            f"Energy: {concept['energy']}/5\n"
            f"Mood: {concept['mood']}\n"
            f"Instruments: {', '.join(concept['instruments'])}\n"
            f"Structure: {concept['structure']}\n\n"
            f"Return ONLY the prompt text, optimized for Suno/Udio AI music generation. "
            f"Be specific about sounds, textures, dynamics, and arrangement."
        )

        response = await client.messages.create(
            model=settings.llm_model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        return {"prompt": response.content[0].text, "llm_enhanced": True}

    def _build_generation_prompt(self, concept: dict) -> str:
        """Build a structured generation prompt from concept data."""
        instruments_str = ", ".join(concept["instruments"])
        return (
            f"{concept['genre'].replace('_', ' ')}, "
            f"{concept['bpm']} BPM, {concept['key']}, "
            f"energy level {concept['energy']}/5, {concept['mood']}, "
            f"featuring {instruments_str}, "
            f"structure: {concept['structure']}, "
            f"professional studio quality, clean mix, "
            f"suitable for radio broadcast"
        )

    def _generate_mood(self, genre: Genre, energy: int) -> str:
        moods = {
            Genre.DRUM_AND_BASS: ["intense", "dark", "energetic", "powerful"],
            Genre.LIQUID_DNB: ["smooth", "dreamy", "emotional", "flowing"],
            Genre.DUBSTEP_MELODIC: ["cinematic", "epic", "emotional", "heavy"],
            Genre.HOUSE_DEEP: ["groovy", "warm", "hypnotic", "sensual"],
            Genre.HOUSE_PROGRESSIVE: ["driving", "uplifting", "evolving", "powerful"],
            Genre.TRANCE_UPLIFTING: ["euphoric", "epic", "uplifting", "transcendent"],
            Genre.TRANCE_PSY: ["psychedelic", "trippy", "intense", "cosmic"],
            Genre.TECHNO_MELODIC: ["hypnotic", "dark", "atmospheric", "driving"],
            Genre.BREAKBEAT: ["funky", "energetic", "playful", "groovy"],
            Genre.AMBIENT: ["ethereal", "peaceful", "expansive", "meditative"],
            Genre.DOWNTEMPO: ["chill", "relaxed", "warm", "contemplative"],
        }
        return random.choice(moods.get(genre, ["energetic"]))
