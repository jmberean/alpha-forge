# Universal devcontainer with Python 3.11 + Node.js 20 (multi-arch)
# Python 3.11 is the base - no need to install it separately
FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Install Node.js 20
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code
