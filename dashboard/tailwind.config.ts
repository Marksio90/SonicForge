import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        sf: {
          bg: "#0a0a0f",
          surface: "#12121a",
          border: "#1e1e2e",
          accent: "#7c3aed",
          "accent-light": "#a78bfa",
          green: "#22c55e",
          red: "#ef4444",
          yellow: "#eab308",
          cyan: "#06b6d4",
        },
      },
    },
  },
  plugins: [],
};

export default config;
