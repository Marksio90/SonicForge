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

# Keys that work best for specific energy levels
ENERGY_KEY_PREFERENCES = {
    1: ["A minor", "D minor", "C minor", "E minor"],
    2: ["E minor", "G minor", "A minor", "F minor", "D major"],
    3: ["F# minor", "A minor", "C# minor", "G major", "D major"],
    4: ["F minor", "G# minor", "C# minor", "A major", "F major"],
    5: ["C# minor", "D# minor", "F# minor", "G# minor"],
}

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
    Genre.DRUM_AND_BASS: ["rolling breaks", "reese bass", "sub bass", "amen break", "vocal chops", "atmospheric pads", "hi-hat patterns", "snare rolls"],
    Genre.LIQUID_DNB: ["lush pads", "liquid bass", "piano", "female vocals", "strings", "smooth drums", "arpeggiated synths", "reverb tails"],
    Genre.DUBSTEP_MELODIC: ["heavy bass wobble", "emotional chords", "orchestral elements", "vocal chops", "synth leads", "super saws", "pluck arpeggios"],
    Genre.HOUSE_DEEP: ["deep kick", "warm bass", "subtle pads", "organic percussion", "filtered vocals", "rhodes", "shakers", "congas"],
    Genre.HOUSE_PROGRESSIVE: ["layered synths", "driving bass", "progressive arpeggios", "white noise risers", "subtle vocal", "analog lead", "reverse effects"],
    Genre.TRANCE_UPLIFTING: ["supersaw leads", "euphoric pads", "pluck synths", "piano", "epic strings", "female vocal", "acid 303", "gated reverb"],
    Genre.TRANCE_PSY: ["acid bass", "psychedelic leads", "complex percussion", "atmospheric fx", "303 acid line", "tribal drums", "filtered noise sweeps"],
    Genre.TECHNO_MELODIC: ["minimal kick", "melodic synth loop", "atmospheric textures", "subtle percussion", "deep bass", "tape delay", "granular pads"],
    Genre.BREAKBEAT: ["chopped breaks", "funky bass", "stab synths", "vocal samples", "scratches", "turntable fx", "brass stabs"],
    Genre.AMBIENT: ["evolving pads", "granular textures", "field recordings", "reverb trails", "gentle drones", "wind chimes", "distant bells"],
    Genre.DOWNTEMPO: ["lo-fi drums", "warm bass", "mellow keys", "vinyl crackle", "soft synths", "jazzy chords", "tape saturation"],
}


class ComposerAgent(BaseAgent):
    """Creates music concepts, analyzes trends, and crafts generation prompts using OpenAI."""

    def __init__(self):
        super().__init__("composer")
        self._openai_client = None

    def _get_openai_client(self):
        """Lazy-initialize and reuse the OpenAI async client."""
        if self._openai_client is None and settings.openai_api_key:
            from openai import AsyncOpenAI
            self._openai_client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                max_retries=settings.llm_max_retries,
            )
        return self._openai_client

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
            genre = self._weighted_genre_select()

        genre_enum = Genre(genre)
        profile = GENRE_PROFILES[genre_enum]
        bpm = random.randint(*profile["bpm_range"])
        energy = energy or {"low": 2, "medium": 3, "high": 4}.get(profile["energy"], 3)

        # Energy-aware key selection for better musical cohesion
        preferred_keys = ENERGY_KEY_PREFERENCES.get(energy, MUSICAL_KEYS)
        key = random.choice(preferred_keys)

        instruments = random.sample(
            INSTRUMENTATION.get(genre_enum, ["synth", "bass", "drums"]),
            k=min(5, len(INSTRUMENTATION.get(genre_enum, []))),
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

        self.logger.info("concept_created", genre=genre, bpm=bpm, key=key, energy=energy)
        return concept

    async def analyze_trends(self, genre: str | None = None) -> dict:
        """Analyze current music trends using OpenAI for intelligent trend synthesis."""
        self.logger.info("analyzing_trends", genre=genre)

        client = self._get_openai_client()
        if client:
            try:
                response = await client.chat.completions.create(
                    model=settings.llm_model_fast,
                    max_tokens=512,
                    temperature=0.7,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert electronic music trend analyst. "
                                "Return JSON with keys: trending_bpm (int), "
                                "trending_keys (list of 3 strings), "
                                "trending_elements (list of 3 strings), "
                                "insight (short string)."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Analyze current trends for "
                                f"{'electronic music' if not genre else genre.replace('_', ' ')} "
                                f"in 2025. What BPM, keys, and production elements are trending?"
                            ),
                        },
                    ],
                    response_format={"type": "json_object"},
                )
                import json
                trend_data = json.loads(response.choices[0].message.content)
                trend_data["source"] = "openai_analysis"
                trend_data["genre"] = genre
                return trend_data
            except Exception as e:
                self.logger.warning("trend_analysis_llm_failed", error=str(e))

        return {
            "trending_bpm": random.randint(120, 180),
            "trending_keys": random.sample(MUSICAL_KEYS, 3),
            "trending_elements": ["atmospheric pads", "vocal chops", "rolling bass"],
            "source": "fallback",
            "genre": genre,
        }

    async def craft_prompt(self, concept: dict) -> dict:
        """Use OpenAI to craft an ultra-detailed generation prompt."""
        client = self._get_openai_client()
        if client:
            return await self._llm_craft_prompt(concept)
        return {"prompt": self._build_generation_prompt(concept)}

    async def _llm_craft_prompt(self, concept: dict) -> dict:
        """Use OpenAI GPT-4o to create an expert-level music generation prompt."""
        client = self._get_openai_client()
        if not client:
            return {"prompt": self._build_generation_prompt(concept)}

        system_prompt = (
            "You are an expert electronic music producer with 20+ years of experience. "
            "Create ultra-detailed prompts for AI music generation that capture the essence "
            "of professional productions. Include specific production techniques, sound design "
            "details, mixing approaches, arrangement ideas, and emotional arc. "
            "Your prompts should be optimized for Suno and Udio AI music generation engines. "
            "Return ONLY the prompt text, no explanations."
        )

        user_prompt = (
            f"Create a detailed music generation prompt for:\n"
            f"Genre: {concept['genre'].replace('_', ' ')}\n"
            f"BPM: {concept['bpm']}\n"
            f"Key: {concept['key']}\n"
            f"Energy: {concept['energy']}/5\n"
            f"Mood: {concept['mood']}\n"
            f"Instruments: {', '.join(concept['instruments'])}\n"
            f"Structure: {concept['structure']}\n\n"
            f"Be specific about sounds, textures, dynamics, spatial positioning, "
            f"and arrangement. Include mixing and mastering notes."
        )

        try:
            response = await client.chat.completions.create(
                model=settings.llm_model,
                max_tokens=1024,
                temperature=settings.llm_temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return {"prompt": response.choices[0].message.content, "llm_enhanced": True}
        except Exception as e:
            self.logger.warning("llm_prompt_craft_failed", error=str(e))
            return {"prompt": self._build_generation_prompt(concept)}

    def _build_generation_prompt(self, concept: dict) -> str:
        """Build a structured generation prompt from concept data."""
        instruments_str = ", ".join(concept["instruments"])
        return (
            f"{concept['genre'].replace('_', ' ')}, "
            f"{concept['bpm']} BPM, {concept['key']}, "
            f"energy level {concept['energy']}/5, {concept['mood']}, "
            f"featuring {instruments_str}, "
            f"structure: {concept['structure']}, "
            f"professional studio quality, clean mix, wide stereo image, "
            f"broadcast-ready mastering, suitable for 24/7 radio stream"
        )

    def _generate_mood(self, genre: Genre, energy: int) -> str:
        """Select mood weighted by energy level for better musical coherence."""
        moods = {
            Genre.DRUM_AND_BASS: ["intense", "dark", "energetic", "powerful", "relentless"],
            Genre.LIQUID_DNB: ["smooth", "dreamy", "emotional", "flowing", "ethereal"],
            Genre.DUBSTEP_MELODIC: ["cinematic", "epic", "emotional", "heavy", "soaring"],
            Genre.HOUSE_DEEP: ["groovy", "warm", "hypnotic", "sensual", "intimate"],
            Genre.HOUSE_PROGRESSIVE: ["driving", "uplifting", "evolving", "powerful", "euphoric"],
            Genre.TRANCE_UPLIFTING: ["euphoric", "epic", "uplifting", "transcendent", "heavenly"],
            Genre.TRANCE_PSY: ["psychedelic", "trippy", "intense", "cosmic", "mystical"],
            Genre.TECHNO_MELODIC: ["hypnotic", "dark", "atmospheric", "driving", "meditative"],
            Genre.BREAKBEAT: ["funky", "energetic", "playful", "groovy", "raw"],
            Genre.AMBIENT: ["ethereal", "peaceful", "expansive", "meditative", "celestial"],
            Genre.DOWNTEMPO: ["chill", "relaxed", "warm", "contemplative", "nostalgic"],
        }
        genre_moods = moods.get(genre, ["energetic"])
        if len(genre_moods) > 2 and energy >= 4:
            return random.choice(genre_moods[2:])
        elif len(genre_moods) > 2 and energy <= 2:
            return random.choice(genre_moods[:2])
        return random.choice(genre_moods)

    def _weighted_genre_select(self) -> str:
        """Select genre using weighted random for better variety and quality."""
        genres = list(Genre)
        weights = [GENRE_PROFILES[g].get("weight", 1.0) for g in genres]
        selected = random.choices(genres, weights=weights, k=1)[0]
        return selected.value
