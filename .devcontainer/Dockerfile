FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    software-properties-common \
    gnupg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Node.js 22.x (LTS) via NodeSource ────────────────────────────────────────
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Claude Code (native binary installer — npm is deprecated) ─────────────────
RUN curl -fsSL https://claude.ai/install.sh | bash
ENV PATH="/root/.local/bin:/root/.claude/bin:$PATH"

# ── OpenAI Codex CLI ──────────────────────────────────────────────────────────
RUN npm i -g @openai/codex

# ── Google Gemini CLI ─────────────────────────────────────────────────────────
RUN npm install -g @google/gemini-cli

# ── Python 3.13 via deadsnakes PPA ───────────────────────────────────────────
RUN add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.13 \
    python3.13-venv \
    python3.13-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Bootstrap pip for Python 3.13, then install Modal SDK
RUN python3.13 -m ensurepip --upgrade \
    && python3.13 -m pip install --upgrade pip \
    && python3.13 -m pip install modal