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

### Backend

Create container to hold the library for all backend functionality.

1.

2. Build the backend image with 
```sh
DOCKER_BUILDKIT=1 docker build --ssh default --no-cache -t jules-backend:latest -f backend/Dockerfile .
```

Or with `docker compose build`
```sh
DOCKER_BUILDKIT=1 docker compose build --no-cache backend
```

3. To build a test container for jupyter notebook
```sh
DOCKER_BUILDKIT=1 docker build --ssh default --no-cache -t jules-backend-dev:latest -f backend/dev/Dockerfile .
```
or with `docker compose build`
```sh
DOCKER_BUILDKIT=1 docker compose build --no-cache backend-dev
```
and then run it with 
```sh
docker run --env-file .env -it --rm -p 8888:8888 -v $(pwd)/ipy:/app/ipy jules-backend-dev:latest
```


### API Service: Building with Private Git Dependencies

If your backend or API depends on private git repositories (e.g., via `git+ssh://`), you must use Docker BuildKit with SSH forwarding:

1. Ensure your SSH agent is running and has access to the necessary private key:
   ```sh
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_rsa  # or your relevant key
   ```
2. Build the API image with BuildKit and SSH forwarding:
   ```sh
   DOCKER_BUILDKIT=1 docker build --ssh default -f api/Dockerfile -t nautilus-api .
   ```
3. (Optional) Push the image to your registry:
   ```sh
   docker tag nautilus-api your-registry/nautilus-api:latest
   docker push your-registry/nautilus-api:latest
   ```
4. Deploy with Docker Compose (set `ENV=prod` or as needed):
   ```sh
   export ENV=prod
   docker-compose up -d
   ```

- If you use a CI/CD system, ensure it supports Docker BuildKit and SSH agent forwarding.
- Your SSH key is only available during build and is not copied into the image.

## Config Example

See `config/dev.yaml`, `config/stage.yaml`, `config/prod.yaml`

---

## Quick Start

1. Install Docker and Docker Compose
2. Clone repo
3. Run: `ENV=dev docker-compose up --build`
