FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (for psycopg2-binary, etc.)
RUN apt-get update && apt-get install -y build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

# Copy backend and install as local package
COPY ../backend /app/backend

# Copy config files
COPY ../config /config

# Copy API code
COPY . /app

# Install API and backend dependencies
RUN pip install --no-cache-dir .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
