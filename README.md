# Nautilus Jules - Modern Cloud Deployable Product

## Structure

- `backend/` - Python backend (business logic)
- `api/` - FastAPI app (REST endpoints)
- `frontend/` - Angular app
- `config/` - Shared YAML config for all environments
- `scripts/` - Utility and deployment scripts

## Environments

- `dev`, `stage`, `prod` - Set via `ENV` environment variable
- Each service loads config from `config/{ENV}.yaml`

## Local Development

- Use Docker Compose:
  ```sh
  ENV=dev docker-compose up --build
  ```

## Deployment

- Each service is containerized and can be deployed independently (Azure Web App for Containers, AKS, etc.)
- Use `ENV` to select environment

## Config Example

See `config/dev.yaml`, `config/stage.yaml`, `config/prod.yaml`

---

## Quick Start

1. Install Docker and Docker Compose
2. Clone repo
3. Run: `ENV=dev docker-compose up --build`
