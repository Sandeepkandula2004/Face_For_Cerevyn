from fastapi import FastAPI, APIRouter, UploadFile, File, Depends, HTTPException, Form
from sqlalchemy.orm import Session
import cv2
import numpy as np
import requests
import uuid
np.bool = bool
# IMPORTANT: single source of truth
from utils import (
    get_db,
    enroll_employee_face,
    verify_employee_face,
    supabase_storage,
    face_app,
)

router = APIRouter(prefix="/face", tags=["Face Recognition"])

# ==================================================
# ENROLL EMPLOYEE FACE (ADMIN)
# ==================================================
@router.post("/enroll/{employee_id}")
def enroll_face(
    employee_id: int,
    file: UploadFile | None = File(None),
    image_url: str | None = Form(None),
    db: Session = Depends(get_db)
):
    print(f"[ENROLL] Starting enroll for employee {employee_id}")
    
    if not file and not image_url:
        raise HTTPException(status_code=400, detail="Provide either file or image_url")

    # Load bytes either from uploaded file or remote URL
    if image_url:
        try:
            resp = requests.get(image_url, timeout=10)
            resp.raise_for_status()
            file_bytes = resp.content
            filename = f"{uuid.uuid4()}.jpg"
        except Exception as exc:
            print(f"[ENROLL] Failed to fetch image_url: {exc}")
            raise HTTPException(status_code=400, detail=f"Failed to fetch image_url: {exc}") from exc
    else:
        file_bytes = file.file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file upload")
        filename = file.filename or f"{uuid.uuid4()}.bin"

    print(f"[ENROLL] Loaded {len(file_bytes)} bytes, filename: {filename}")

    image = cv2.imdecode(
        np.frombuffer(file_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        print("[ENROLL] Failed to decode image")
        raise HTTPException(status_code=400, detail="Invalid image content")

    # Store original image in 'face' bucket
    try:
        print(f"[ENROLL] Uploading to supabase bucket 'face'...")
        supabase_storage.storage.from_("face").upload(filename, file_bytes)
        public_url = supabase_storage.storage.from_("face").get_public_url(filename)
        print(f"[ENROLL] Uploaded successfully: {public_url}")
    except Exception as exc:
        print(f"[ENROLL] Supabase upload failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to store image: {exc}") from exc

    print(f"[ENROLL] Calling enroll_employee_face()...")
    success = enroll_employee_face(db, employee_id, image, image_url=public_url)

    if not success:
        print(f"[ENROLL] Face enrollment returned False")
        raise HTTPException(
            status_code=400,
            detail="Face not detected or enrollment failed"
        )

    print(f"[ENROLL] Success!")
    return {
        "message": "Employee face enrolled successfully",
        "employee_id": employee_id,
        "image_url": public_url,
    }

# ==================================================
# VERIFY EMPLOYEE FACE (CHECK-IN)
# ==================================================
@router.post("/verify")
def verify_face(
    file: UploadFile = File(...)
):
    file_bytes = file.file.read()

    image = cv2.imdecode(
        np.frombuffer(file_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    employee_id = verify_employee_face(image)

    if employee_id is None:
        return {
            "match": False,
            "confidence": None
        }

    return {
        "match": True,
        "employee_id": employee_id
    }

# ==================================================
# DELETE EMPLOYEE FACE
# ==================================================
@router.delete("/delete/{employee_id}")
def delete_employee_face(
    employee_id: int,
    db: Session = Depends(get_db)
):
    from utils import rebuild_faiss
    from models import EmployeeFace
    
    face = db.query(EmployeeFace).filter(EmployeeFace.employee_id == employee_id).first()
    
    if not face:
        raise HTTPException(status_code=404, detail=f"Employee {employee_id} not found")
    
    # Delete image from bucket if reference_image_url exists
    if face.reference_image_url is not None:
        try:
            # Extract filename from public URL (e.g., https://.../<bucket>/<filename> -> <filename>)
            url_parts = face.reference_image_url.split("/")
            filename = url_parts[-1] if url_parts else None
            
            if filename:
                supabase_storage.storage.from_("face").remove([filename])
        except Exception as exc:  # noqa: BLE001
            # Log but don't fail if bucket delete fails
            print(f"Warning: Failed to delete image from bucket: {exc}")
    
    db.delete(face)
    db.commit()
    
    # Rebuild FAISS index after deletion
    rebuild_faiss(db)
    
    return {
        "message": "Employee face deleted successfully",
        "employee_id": employee_id
    }

# ==================================================
# ASGI APP
# ==================================================
app = FastAPI(title="Face Service")
app.include_router(router)

# ==================================================
# STARTUP: Warm up InsightFace
# ==================================================
@app.on_event("startup")
def warmup_face_model():
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    try:
        face_app.get(dummy)
        print("🔥 InsightFace warmed up")
    except Exception as exc:  # noqa: BLE001
        print(f"InsightFace warmup skipped/failed: {exc}")

