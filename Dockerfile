# Dockerfile
FROM python:3.11-slim

# System deps (optional, keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
 && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -u 10001 agent
USER agent

WORKDIR /app

# Use a non-interactive backend for matplotlib
ENV MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy only deps first for better layer caching
COPY --chown=agent:agent requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY --chown=agent:agent sql_pandas_agent.py .

# tini as PID 1 for clean signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash"]