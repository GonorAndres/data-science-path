import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        paper: "#FAFAF8",
        ink: "#1A1A1A",
        "glm-blue": "#1E3A5F",
        "gbm-green": "#2D6B45",
        "lgbm-teal": "#1A7A6D",
        "accent-amber": "#C4841D",
      },
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        serif: ["var(--font-playfair)", "Georgia", "serif"],
      },
      maxWidth: {
        "5xl": "64rem",
        "2xl": "42rem",
      },
    },
  },
  plugins: [],
};

export default config;
