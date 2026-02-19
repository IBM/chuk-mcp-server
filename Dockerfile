# ChukMCPServer Dockerfile
# Multi-stage build for optimized production deployment

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM python:3.14-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /build

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY README.md ./

# Install dependencies and build
RUN uv pip install --system -e .

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM python:3.14-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for npx-based servers (optional)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /build/src /app/src
COPY --from=builder /build/pyproject.toml /app/

# Copy configuration files
COPY config.yaml /app/config.yaml
COPY .env.example /app/.env.example

# Create directories for modules and data
RUN mkdir -p /app/modules /app/data && chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:${PATH}"

# Expose default HTTP port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: run HTTP server with config
CMD ["python", "-m", "chuk_mcp_server.cli", "--config", "config.yaml"]
