# Nautilus API

This FastAPI backend exposes the core functionality of the Nautilus Jules system to the Angular frontend. It wraps the `jules_core` backend package and provides RESTful endpoints for models, operability, RAO, cables, vessels, and authentication.

## API Endpoints

### Models
- `GET /api/v1/models/cables` — List available cable types
- `POST /api/v1/models/cables/{cable_type}/predict` — Predict outcomes for given features

### Operability
- `POST /api/v1/operability/heatmap` — Get operability heatmap
- `POST /api/v1/operability/matrix` — Get operability matrix

### RAO
- `POST /api/v1/raos` — Calculate RAO values
- `POST /api/v1/raos/plot` — Generate RAO plot

### Cables
- `GET /api/v1/cables` — List available cables
- `POST /api/v1/cables/{cable_type}` — Get cable record

### Vessels
- `GET /api/v1/vessels` — List available vessels
- `POST /api/v1/vessels/types` — List vessel types for a vessel
- `POST /api/v1/vessels/draughts` — List draughts for a vessel and type

### Authentication
- `POST /api/v1/auth/signup` — User registration
- `POST /api/v1/auth/login` — User login

## Usage Examples

### List Cable Types
```bash
curl -X GET "http://localhost:8000/api/v1/models/cables"
```

### Predict Cable Outcomes
```bash
curl -X POST "http://localhost:8000/api/v1/models/cables/EXC1000AL/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature1": 1.0, "feature2": 2.0}}'
```

### User Signup
```bash
curl -X POST "http://localhost:8000/api/v1/auth/signup" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "string", "full_name": "User Name"}'
```

### User Login
```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "string"}'
```

## Documentation

- Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Redoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Development

- Install dependencies: `pip install -r requirements.txt`
- Run the app: `uvicorn app.main:app --reload`

## Docker

- Build: `docker build -t nautilus-api .`
- Run: `docker-compose up`

## License
MIT
