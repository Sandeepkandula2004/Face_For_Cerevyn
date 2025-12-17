# Face Recognition Service

Fast, scalable face enrollment and verification using InsightFace + FAISS + Supabase.

## Features

- ✅ **Face Enrollment**: Upload images or URLs, augment to 12 variants, compute mean embeddings
- ✅ **Face Verification**: 1:1 matching against FAISS index with cosine similarity
- ✅ **Async FAISS Rebuild**: Background threading prevents blocking on index updates
- ✅ **Thread-Safe Embeddings**: Lock serializes embedding extraction across workers
- ✅ **Storage**: Supabase bucket integration with service-role key support
- ✅ **Multi-Worker**: 2-worker Uvicorn setup for high throughput
- ✅ **Startup Warmup**: Pre-loads InsightFace models on app start
- ✅ **React Frontend**: Simple tester UI with enrollment/verification forms
- ✅ **Docker Ready**: Python 3.12 slim, production-optimized

## Stack

- **Backend**: FastAPI + Uvicorn
- **Face AI**: InsightFace (AntlopeV2) with 512-dim embeddings
- **Search**: FAISS (IndexFlatIP, cosine similarity)
- **Database**: Supabase PostgreSQL
- **Storage**: Supabase Storage (face bucket)
- **Frontend**: React + Vite
- **Deployment**: Docker + HuggingFace Spaces ready

## Quick Start

### Local Development

1. Clone & install:
```bash
git clone <repo>
cd Face_for_Cerevyn
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Set environment (`.env`):
```
SUPABASE_PROJECT_URL=https://your-project.supabase.co
ANON_KEY=your_anon_key
SUPABASE_DB_URL=postgresql://...
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key  # optional, for storage delete
```

3. Run backend (port 7000):
```bash
uvicorn app:app --host 127.0.0.1 --port 7000 --reload
```

4. Run frontend (from `my-react-app/`):
```bash
npm install
npm run dev
```

5. Visit http://localhost:5173 to test enrollment/verification.

### Docker

Build & run:
```bash
docker build -t face-service:latest .
docker run --env-file .env -p 7000:7000 face-service:latest
```

### Production (HuggingFace Spaces)

1. Create a new Space with Docker runtime
2. Upload repo or connect GitHub
3. In **Settings → Secrets**, add:
   - `SUPABASE_PROJECT_URL`
   - `ANON_KEY`
   - `SUPABASE_DB_URL`
   - `SUPABASE_SERVICE_ROLE_KEY`
4. HF will auto-build & deploy

## API Routes

### Enroll Face
```http
POST /face/enroll/{employee_id}
Content-Type: multipart/form-data

file: <image>  # OR
image_url: https://...
```

Response: `{ "message": "...", "employee_id": 11, "image_url": "https://..." }`

### Verify Face
```http
POST /face/verify
Content-Type: multipart/form-data

file: <image>
```

Response: `{ "match": true, "employee_id": 11 }` or `{ "match": false }`

### Delete Employee
```http
DELETE /face/delete/{employee_id}
```

Response: `{ "message": "Employee face deleted successfully", "employee_id": 11 }`

## Configuration

- **Augmentation count**: 12 (fast), in `utils.enroll_employee_face()`
- **Verification threshold**: 0.35 (cosine similarity), in `utils.verify_employee_face()`
- **Workers**: 2, in `Dockerfile` CMD
- **Lock**: Serializes embedding extraction per-process

## Architecture

```
FastAPI (app.py)
  ├─ /enroll → enroll_employee_face() → augment → embedding → FAISS rebuild (async)
  ├─ /verify → extract_embedding() → FAISS search → employee_id
  └─ /delete → remove from DB & Supabase bucket

utils.py
  ├─ face_app (InsightFace, warmed up on startup)
  ├─ augment_image (imgaug, 12 variants)
  ├─ extract_embedding (serialized by face_lock)
  └─ rebuild_faiss (background thread, fresh session)

supabase_storage (service-role key for uploads/deletes)
```

## Testing

Standalone tests in `test.py`:
```bash
python test.py enroll    # Interactive webcam enrollment
python test.py verify    # Interactive webcam verification
```

## License

MIT
