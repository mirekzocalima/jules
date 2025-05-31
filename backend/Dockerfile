# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY config ./config
COPY backend/pyproject.toml ./
COPY backend/src ./src

RUN mkdir -p /root/.ssh && ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN --mount=type=ssh \
    /bin/sh -c "export SSH_AUTH_SOCK=$SSH_AUTH_SOCK && pip install --upgrade pip && pip install ."