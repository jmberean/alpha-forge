#!/bin/bash

# AlphaForge - Development Startup (with terminal output)
# Shows real-time logs in the terminal using tmux

set -e

# Get absolute path to project root
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "=================================="
echo "   ALPHAFORGE MVP8 STARTUP"
echo "=================================="
echo ""

# Check if tmux is available
if ! command -v tmux &> /dev/null; then
    echo "⚠ tmux not found. Installing..."
    sudo apt-get update && sudo apt-get install -y tmux
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "⚠ Virtual environment not found. Creating..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "Installing dependencies..."
    pip install -e ".[dev]"
else
    source .venv/bin/activate
fi

# Check if node_modules exists
if [ ! -d "frontend/node_modules" ]; then
    echo "⚠ Node modules not found. Installing..."
    cd frontend
    npm install
    cd "$PROJECT_ROOT"
fi

echo "✓ Environment ready"
echo ""
echo "Starting services in tmux session 'alphaforge'..."
echo ""

# Kill existing session if it exists
tmux kill-session -t alphaforge 2>/dev/null || true

# Create new tmux session
tmux new-session -d -s alphaforge -n "AlphaForge"

# Split window vertically
tmux split-window -h -t alphaforge

# Backend in left pane
tmux send-keys -t alphaforge:0.0 "source .venv/bin/activate" C-m
tmux send-keys -t alphaforge:0.0 "cd src && python -m alphaforge.api.server" C-m

# Frontend in right pane
tmux send-keys -t alphaforge:0.1 "cd frontend && npm run dev" C-m

# Set pane titles
tmux select-pane -t alphaforge:0.0 -T "Backend API"
tmux select-pane -t alphaforge:0.1 -T "Frontend"

echo "✓ Services started!"
echo ""
echo "=================================="
echo "   ALPHAFORGE IS RUNNING"
echo "=================================="
echo ""
echo "Services:"
echo "  • Frontend:  http://localhost:3000"
echo "  • Backend:   http://localhost:8000"
echo "  • API Docs:  http://localhost:8000/docs"
echo ""
echo "Tmux session: 'alphaforge'"
echo ""
echo "Commands:"
echo "  • Attach to session:  tmux attach -t alphaforge"
echo "  • Stop all services:  ./stop.sh"
echo "  • Switch panes:       Ctrl+b then arrow keys"
echo "  • Detach:             Ctrl+b then d"
echo ""

# Attach to the session
tmux attach -t alphaforge
