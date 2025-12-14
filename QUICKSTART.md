# AlphaForge Quick Start

## Setup

```bash
# Install Python dependencies
source .venv/bin/activate

# Install frontend dependencies
cd frontend && npm install && cd ..
```

## Run

**Terminal 1 - Backend:**
```bash
source .venv/bin/activate
cd src && python -m alphaforge.api.server
```

**Terminal 2 - Frontend:**
```bash
cd frontend && npm run dev
```

**Open:** http://localhost:3000

## Usage

1. Click **START FACTORY** to generate strategies via genetic evolution
2. Watch strategies get validated (DSR, CPCV, Stress tests)
3. View results in Strategy Candidates list
4. Or use **Single Strategy Validation** to test individual templates

## API Docs

http://localhost:8000/docs
