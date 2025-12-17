import os
import uuid
from threading import Lock, Thread
import cv2
import faiss
import numpy as np
import imgaug.augmenters as iaa

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from supabase import create_client
from insightface.app import FaceAnalysis
from models import EmployeeFace

# ==================================================
# ENV + SUPABASE
# ==================================================
# For local dev: uses .env file via load_dotenv()
# For HuggingFace Spaces: reads from Settings > Secrets (auto-exposed as env vars)
load_dotenv()

SUPABASE_PROJECT_URL = os.getenv("SUPABASE_PROJECT_URL")
SUPABASE_ANON_KEY = os.getenv("ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

if not SUPABASE_PROJECT_URL:
    raise ValueError("SUPABASE_PROJECT_URL environment variable must be set")
if not SUPABASE_ANON_KEY:
    raise ValueError("ANON_KEY environment variable must be set (set in HF Spaces Secrets)")
if not SUPABASE_DB_URL:
    raise ValueError("SUPABASE_DB_URL environment variable must be set (set in HF Spaces Secrets)")

# Use service-role key for storage if provided, fall back to anon
_storage_key = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY
supabase_storage = create_client(SUPABASE_PROJECT_URL, _storage_key)
# Keep anon client available if needed elsewhere
supabase = create_client(SUPABASE_PROJECT_URL, SUPABASE_ANON_KEY)

# ==================================================
# DATABASE (UNCHANGED)
# ==================================================
engine = create_engine(
    SUPABASE_DB_URL,
    pool_pre_ping=True,
    pool_recycle=1800,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==================================================
# IMAGE UPLOAD (UNCHANGED)
# ==================================================
def upload_to_bucket(file, bucket_name: str):
    file_bytes = file.file.read()
    ext = file.filename.split(".")[-1]
    file_name = f"{uuid.uuid4()}.{ext}"

    supabase_storage.storage.from_(bucket_name).upload(file_name, file_bytes)
    return supabase_storage.storage.from_(bucket_name).get_public_url(file_name)

def upload_selfie(file):
    return upload_to_bucket(file, "selfies")

# ==================================================
# FACE RECOGNITION CONFIG (test_enroll style)
# ==================================================
EMBEDDING_DIM = 512
FAISS_INDEX_PATH = "employee_faces.faiss"
FAISS_MAP_PATH = "employee_faces_map.npy"

face_app = FaceAnalysis(
    name="antelopev2",
    providers=["CPUExecutionProvider"]
)
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Serialize embedding computation to avoid race conditions
face_lock = Lock()

# ==================================================
# AUGMENTATION (SAME PHILOSOPHY AS test_enroll)
# ==================================================
augmenter = iaa.SomeOf((2, 4), [
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-45, 45)),
    iaa.Multiply((0.8, 1.2)),
    iaa.GaussianBlur(sigma=(0, 1.0)),
    iaa.AdditiveGaussianNoise(scale=(10, 30)),
    iaa.Sharpen(alpha=(0.2, 0.5), lightness=(0.8, 1.2)),
    iaa.Crop(percent=(0, 0.1)),
    iaa.LinearContrast((0.75, 1.5)),
    iaa.SomeOf((0, 1), [iaa.Grayscale(alpha=1.0)])
])

def augment_image(image_rgb, count=50):
    return augmenter(images=[image_rgb] * count)

# ==================================================
# EMBEDDING EXTRACTION
# ==================================================
def extract_embedding(image_rgb):
    faces = face_app.get(image_rgb)
    if not faces:
        return None

    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )
    return face.normed_embedding.astype("float32")

def mean_embedding(images):
    embeddings = []

    for img in images:
        emb = extract_embedding(img)
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        return None

    mean_emb = np.mean(embeddings, axis=0)
    mean_emb /= np.linalg.norm(mean_emb)
    return mean_emb.astype("float32")

# ==================================================
# FAISS + MAPPING (IMPORTANT)
# ==================================================
def rebuild_faiss(db: Session):
    """
    Builds FAISS index AND employee_id mapping
    """
    faces = db.query(EmployeeFace).order_by(EmployeeFace.employee_id).all()
    if not faces:
        return

    vectors = []
    id_map = []

    for f in faces:
        vectors.append(f.embedding)
        id_map.append(f.employee_id)

    vectors = np.array(vectors, dtype="float32")

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)

    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(FAISS_MAP_PATH, np.array(id_map))

def load_faiss():
    if not os.path.exists(FAISS_INDEX_PATH):
        return None, None

    index = faiss.read_index(FAISS_INDEX_PATH)
    id_map = np.load(FAISS_MAP_PATH)

    return index, id_map

# ==================================================
# FACE ENROLLMENT (DB + FAISS)
# ==================================================
def enroll_employee_face(db: Session, employee_id: int, image_bgr, image_url: str = None):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    augmented = augment_image(image_rgb, 12)
    with face_lock:
        mean_emb = mean_embedding(augmented)

    if mean_emb is None:
        return False

    face = (
        db.query(EmployeeFace)
        .filter(EmployeeFace.employee_id == employee_id)
        .first()
    )

    if face:
        face.embedding = mean_emb.tolist()
        if image_url:
            face.reference_image_url = image_url
    else:
        face = EmployeeFace(
            employee_id=employee_id,
            embedding=mean_emb.tolist(),
            reference_image_url=image_url
        )
        db.add(face)

    db.commit()

    # Rebuild FAISS in background using a fresh session to avoid closed-session issues
    def _rebuild():
        _db = SessionLocal()
        try:
            rebuild_faiss(_db)
        finally:
            _db.close()

    Thread(target=_rebuild, daemon=True).start()
    return True

# ==================================================
# FACE VERIFICATION (WITH MAPPING)
# ==================================================
def verify_employee_face(image_bgr, threshold=0.35):
    index, id_map = load_faiss()
    if index is None:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    emb = extract_embedding(image_rgb)

    if emb is None:
        return None

    emb = emb.reshape(1, -1).astype("float32")
    D, I = index.search(emb, 1)

    score = float(D[0][0])
    if score >= threshold:
        return int(id_map[I[0][0]])

    return None
