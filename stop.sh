#!/bin/bash

# AlphaForge - Stop all services

echo "Stopping AlphaForge services..."

# Kill tmux session if it exists
if tmux has-session -t alphaforge 2>/dev/null; then
    tmux kill-session -t alphaforge
    echo "✓ Stopped tmux session 'alphaforge'"
fi

# Kill any remaining backend/frontend processes
pkill -f "alphaforge.api.server" 2>/dev/null && echo "✓ Stopped backend API" || true
pkill -f "next-server" 2>/dev/null && echo "✓ Stopped frontend dev server" || true

echo ""
echo "All services stopped."
