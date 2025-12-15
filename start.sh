#!/bin/bash

# AlphaForge - Development Environment Startup Script
# Starts both backend API and frontend dev server

set -e

# Get absolute path to project root
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "=================================="
echo "   ALPHAFORGE MVP8 STARTUP"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create logs directory if it doesn't exist
mkdir -p logs

# Setup virtual environment if needed
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚠ Virtual environment not found. Creating...${NC}"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"
else
    source .venv/bin/activate
fi

# Use venv python
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"

# Check if node_modules exists
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${YELLOW}⚠ Node modules not found. Installing...${NC}"
    cd frontend
    npm install
    cd "$PROJECT_ROOT"
fi

echo -e "${GREEN}✓ Environment ready${NC}"
echo ""

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down services...${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo -e "${GREEN}✓ Services stopped${NC}"
    exit 0
}

trap cleanup INT TERM

# Start backend
echo -e "${CYAN}[1/2] Starting Backend API...${NC}"
cd "$PROJECT_ROOT/src"
"$PYTHON_BIN" -m alphaforge.api.server > "$PROJECT_ROOT/logs/backend.log" 2>&1 &
BACKEND_PID=$!
cd "$PROJECT_ROOT"

# Wait a moment for backend to start
sleep 2

# Check if backend is running
if kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${GREEN}✓ Backend API running on http://localhost:8000${NC}"
    echo -e "${CYAN}  API Docs: http://localhost:8000/docs${NC}"
else
    echo -e "${YELLOW}⚠ Backend failed to start. Check logs/backend.log${NC}"
    exit 1
fi

# Start frontend
echo ""
echo -e "${CYAN}[2/2] Starting Frontend...${NC}"
cd "$PROJECT_ROOT/frontend"
npm run dev > "$PROJECT_ROOT/logs/frontend.log" 2>&1 &
FRONTEND_PID=$!
cd "$PROJECT_ROOT"

# Wait a moment for frontend to start
sleep 3

# Check if frontend is running
if kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${GREEN}✓ Frontend running on http://localhost:3000${NC}"
else
    echo -e "${YELLOW}⚠ Frontend failed to start. Check logs/frontend.log${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

echo ""
echo "=================================="
echo -e "${GREEN}   ✓ ALPHAFORGE IS RUNNING${NC}"
echo "=================================="
echo ""
echo "Services:"
echo "  • Frontend:  http://localhost:3000"
echo "  • Backend:   http://localhost:8000"
echo "  • API Docs:  http://localhost:8000/docs"
echo ""
echo "Logs:"
echo "  • Backend:   logs/backend.log"
echo "  • Frontend:  logs/frontend.log"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Tail logs to show activity
tail -f "$PROJECT_ROOT/logs/backend.log" "$PROJECT_ROOT/logs/frontend.log"
