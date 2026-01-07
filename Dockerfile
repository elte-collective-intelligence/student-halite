# Dockerfile for Halite Multi-Agent Reinforcement Learning
# Uses Python slim image for smaller size
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements with pinned versions
COPY requirements-lock.txt /app/requirements-lock.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-lock.txt

# Copy project files
COPY . /app

# Set Python path
ENV PYTHONPATH=/app

# Default command (can be overridden)
CMD ["python", "--version"]



