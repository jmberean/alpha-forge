import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        terminal: {
          bg: '#0a0e14',
          panel: '#0f1419',
          border: '#1a1f29',
          text: '#b3b8c4',
          bright: '#e6e9f0',
          green: '#7fd962',
          amber: '#ffb454',
          red: '#f07178',
          blue: '#59c2ff',
          purple: '#d2a6ff',
          cyan: '#95e6cb',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        display: ['Orbitron', 'monospace'],
        sans: ['IBM Plex Sans', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'scan': 'scan 4s linear infinite',
        'glow': 'glow 2s ease-in-out infinite',
      },
      keyframes: {
        scan: {
          '0%, 100%': { opacity: '0.2' },
          '50%': { opacity: '0.8' },
        },
        glow: {
          '0%, 100%': {
            boxShadow: '0 0 5px rgba(127, 217, 98, 0.3), 0 0 10px rgba(127, 217, 98, 0.2)'
          },
          '50%': {
            boxShadow: '0 0 10px rgba(127, 217, 98, 0.5), 0 0 20px rgba(127, 217, 98, 0.3)'
          },
        },
      },
    },
  },
  plugins: [],
}
export default config
