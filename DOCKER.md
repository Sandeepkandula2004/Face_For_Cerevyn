# Face Recognition Service Dockerfile

Created `Dockerfile` with Python 3.12 slim base image.

## Features
- Python 3.12 slim (minimal footprint)
- Installs system dependencies for OpenCV/InsightFace (libsm6, libxext6, libgomp1)
- Copies requirements.txt first for Docker layer caching
- Runs Uvicorn on 0.0.0.0:7000 with 2 workers
- Health check to ensure app is responsive
- Non-root best practices via Python image defaults

## Build & Run

Build the image:
```bash
docker build -t face-service:latest .
```

Run a container:
```bash
docker run \
  --env SUPABASE_PROJECT_URL="<url>" \
  --env ANON_KEY="<anon_key>" \
  --env SUPABASE_DB_URL="<db_url>" \
  --env SUPABASE_SERVICE_ROLE_KEY="<service_role_key>" \
  -p 7000:7000 \
  face-service:latest
```

Or with a `.env` file:
```bash
docker run --env-file .env -p 7000:7000 face-service:latest
```

## Notes
- Change `--workers 2` in `CMD` to match your CPU cores
- Requires all 4 Supabase env vars set (or it will fail to start)
- Health check validates the `/docs` endpoint is reachable
