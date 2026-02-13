"use client";

const GENRES = [
  { value: "drum_and_bass", label: "Drum & Bass", icon: "🥁", color: "#ff6600" },
  { value: "liquid_dnb", label: "Liquid DnB", icon: "💧", color: "#4488ff" },
  { value: "dubstep_melodic", label: "Melodic Dubstep", icon: "🔊", color: "#ff0066" },
  { value: "house_deep", label: "Deep House", icon: "🏠", color: "#ff8844" },
  { value: "house_progressive", label: "Progressive House", icon: "📈", color: "#6644ff" },
  { value: "trance_uplifting", label: "Uplifting Trance", icon: "✨", color: "#0088ff" },
  { value: "trance_psy", label: "Psytrance", icon: "🌀", color: "#ff00ff" },
  { value: "techno_melodic", label: "Melodic Techno", icon: "⚡", color: "#00aaff" },
  { value: "breakbeat", label: "Breakbeat", icon: "💥", color: "#ff6600" },
  { value: "ambient", label: "Ambient", icon: "🌿", color: "#228844" },
  { value: "downtempo", label: "Downtempo", icon: "🌙", color: "#886644" },
];

interface GenreSelectorProps {
  selected: string | null;
  onSelect: (genre: string) => void;
}

export default function GenreSelector({ selected, onSelect }: GenreSelectorProps) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
      {GENRES.map((genre) => (
        <button
          key={genre.value}
          onClick={() => onSelect(genre.value)}
          className={`flex items-center gap-2 px-3 py-2.5 rounded-lg border text-sm transition-all ${
            selected === genre.value
              ? "border-sf-accent bg-sf-accent/20 text-white"
              : "border-sf-border bg-sf-bg text-gray-400 hover:border-sf-accent/50 hover:text-gray-200"
          }`}
        >
          <span>{genre.icon}</span>
          <span>{genre.label}</span>
        </button>
      ))}
    </div>
  );
}
