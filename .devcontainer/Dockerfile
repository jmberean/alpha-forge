# .devcontainer/Dockerfile
FROM mcr.microsoft.com/devcontainers/javascript-node:1-20-bullseye

# Install the AI tools during the BUILD phase (saved in the image)
# This means they are instant when you start the container
RUN npm install -g @anthropic-ai/claude-code @google/gemini-cli