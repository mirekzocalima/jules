# syntax=docker/dockerfile:1.2

FROM python:3.11-slim

LABEL maintainer="Damir Herman <damir@nautilus-ai.org>"
LABEL org.opencontainers.image.source="https://github.com/nautilus-ai/jules"
LABEL org.opencontainers.image.version="0.0.1"

WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY jules_backend ./jules_backend

RUN mkdir -p /root/.ssh && ssh-keyscan github.com >> /root/.ssh/known_hosts
RUN --mount=type=ssh pip install --upgrade pip && pip install .