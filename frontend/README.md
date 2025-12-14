# AlphaForge Frontend

**Quantitative Terminal Aesthetic** - Production-grade dashboard for systematic trading strategy validation.

## Design Philosophy

This frontend embodies a **dark terminal aesthetic** inspired by professional trading terminals, blending precision with modern design:

- **Typography**: Orbitron (display), JetBrains Mono (metrics/code), IBM Plex Sans (body)
- **Color Palette**: Dark terminal theme with green/amber accents (classic trading colors)
- **Motion**: Framer Motion animations for data reveals and interactions
- **Atmosphere**: Grid background, scanline effects, terminal-style logging

## Key Features

### 1. Validation Pipeline Visualization
Shows the brutal funnel from 10,000 strategy candidates → deployment (0.03% survival rate)
- Real-time progress bars
- Stage-by-stage metrics (DSR, CPCV, PBO, event-driven, SPA)
- Visual degradation flow

### 2. Metrics Dashboard
- 8 key validation metrics with thresholds
- Real-time status indicators
- Color-coded pass/fail states

### 3. Equity Curve Chart
- Interactive Recharts visualization
- Strategy vs benchmark comparison
- Trade statistics panel

### 4. Strategy Candidates List
- Top 6 strategies from validation pipeline
- Expandable details per strategy
- Status tracking (deployed/approved/testing/rejected)

### 5. System Terminal
- Live command log with timestamp
- Interactive command input
- Simulated AlphaForge CLI

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS + Custom CSS
- **Animation**: Framer Motion
- **Charts**: Recharts
- **Fonts**: Google Fonts (Orbitron, JetBrains Mono, IBM Plex Sans)

## Installation

```bash
# Install dependencies
npm install
# or
yarn install
# or
pnpm install

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the dashboard.

## Integration with Backend

The frontend is designed to connect to the AlphaForge Python backend via REST API:

```typescript
// Example API integration
const validateStrategy = async (symbol: string, template: string) => {
  const response = await fetch('http://localhost:8000/api/validate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbol, template, n_trials: 1000 }),
  })
  return response.json()
}
```

### API Endpoints (To be implemented)

- `POST /api/validate` - Run strategy validation
- `GET /api/strategies` - List strategy candidates
- `GET /api/backtest/:id` - Get backtest results
- `GET /api/metrics/:strategy_id` - Get validation metrics
- `POST /api/generate` - Generate new strategies

## Customization

### Theme Colors

Edit `tailwind.config.ts` to customize the terminal color palette:

```typescript
colors: {
  terminal: {
    bg: '#0a0e14',      // Main background
    panel: '#0f1419',   // Panel background
    green: '#7fd962',   // Success/profit
    amber: '#ffb454',   // Warning/neutral
    red: '#f07178',     // Error/loss
  }
}
```

### Grid and Scanline Effects

Toggle effects in `app/globals.css`:

```css
/* Disable grid */
body::before {
  display: none;
}

/* Disable scanlines */
body::after {
  display: none;
}
```

## Production Build

```bash
npm run build
npm start
```

## Design Credits

This interface avoids generic "AI aesthetics" through:
- Custom terminal-inspired design language
- Distinctive font pairing (Orbitron + JetBrains Mono)
- Mathematical precision in layout and spacing
- Ambient effects (grid, scanlines) that enhance atmosphere
- Data-first composition with strategic use of motion

## License

MIT - Part of the AlphaForge project
