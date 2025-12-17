import cv2
import faiss
import numpy as np
import os
import imgaug.augmenters as iaa

from insightface.app import FaceAnalysis
from sqlalchemy.orm import Session

from models import EmployeeFace, Base
from utils import get_db, engine

np.bool = bool 

# ==================================================
# DATABASE SESSION
# ==================================================
db: Session = next(get_db())
Base.metadata.create_all(bind=engine)

# ==================================================
# ArcFace (same as util.py)
# ==================================================
face_app = FaceAnalysis(
    name="antelopev2",
    providers=["CPUExecutionProvider"]
)
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ==================================================
# CONFIG
# ==================================================
EMBEDDING_DIM = 512
FAISS_INDEX_PATH = "employee_faces.faiss"
FAISS_MAP_PATH = "employee_faces_map.npy"
VERIFY_THRESHOLD = 0.40  # cosine similarity

# ==================================================
# AUGMENTATION (same philosophy as test_enroll)
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

# ==================================================
# EMBEDDING HELPERS
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
# FAISS (COSINE + MAPPING)
# ==================================================
def rebuild_faiss():
    faces = db.query(EmployeeFace).order_by(EmployeeFace.employee_id).all()
    if not faces:
        print("❌ No employee faces in DB")
        return None, None

    vectors = np.array([f.embedding for f in faces], dtype="float32")
    id_map = np.array([f.employee_id for f in faces], dtype="int64")

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)

    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(FAISS_MAP_PATH, id_map)

    print(f"✅ FAISS rebuilt with {len(id_map)} employees")
    return index, id_map

def load_faiss():
    if not os.path.exists(FAISS_INDEX_PATH):
        return None, None

    index = faiss.read_index(FAISS_INDEX_PATH)
    id_map = np.load(FAISS_MAP_PATH)
    return index, id_map

# ==================================================
# ENROLLMENT
# ==================================================
def enroll_face(employee_id: int, image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    augmented = augmenter(images=[image_rgb] * 50)
    mean_emb = mean_embedding(augmented)

    if mean_emb is None:
        raise RuntimeError("❌ No face detected during enrollment")

    record = (
        db.query(EmployeeFace)
        .filter(EmployeeFace.employee_id == employee_id)
        .first()
    )

    if record:
        record.embedding = mean_emb.tolist()
    else:
        record = EmployeeFace(
            employee_id=employee_id,
            embedding=mean_emb.tolist()
        )
        db.add(record)

    db.commit()
    print(f"✅ Enrolled employee {employee_id}")

    rebuild_faiss()

# ==================================================
# VERIFICATION
# ==================================================
def verify_face(image_bgr):
    index, id_map = load_faiss()
    if index is None:
        print("❌ FAISS index not found")
        return

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    emb = extract_embedding(image_rgb)

    if emb is None:
        print("❌ No face detected during verification")
        return

    emb = emb.reshape(1, -1).astype("float32")
    D, I = index.search(emb, 1)

    score = float(D[0][0])
    if score >= VERIFY_THRESHOLD:
        employee_id = int(id_map[I[0][0]])
        print(
            f"✅ MATCH: employee_id={employee_id}, similarity={score:.4f}"
        )
    else:
        print(
            f"❌ NO MATCH: similarity={score:.4f}"
        )

# ==================================================
# WEBCAM TESTS
# ==================================================
def test_enroll():
    EMPLOYEE_ID = 11

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open webcam")

    print("▶ Press 's' to capture ENROLL image (ESC to quit)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Camera read failed")

            cv2.imshow("Enroll Capture", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                return
            if key == ord("s"):
                enroll_frame = frame.copy()
                break

        enroll_face(EMPLOYEE_ID, enroll_frame)

    finally:
        cap.release()
        cv2.destroyAllWindows()

def test_verify():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open webcam")

    print("▶ Press 's' to capture VERIFY image (ESC to quit)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Camera read failed")

            cv2.imshow("Verify Capture", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                return
            if key == ord("s"):
                verify_frame = frame.copy()
                break

        verify_face(verify_frame)

    finally:
        cap.release()
        cv2.destroyAllWindows()

# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "enroll":
            test_enroll()
        elif cmd == "verify":
            test_verify()
        else:
            print("Usage: python test_webcam_face.py [enroll|verify]")
    else:
        print("Usage: python test_webcam_face.py [enroll|verify]")

    db.close()
