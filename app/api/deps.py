from typing import Generator
from fastapi import UploadFile
import aiofiles
import os
from sqlalchemy.orm import Session
from app.core.config import settings
from app.db.session import SessionLocal

def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def save_upload_file(upload_file: UploadFile) -> str:
    """Simpan file upload ke storage"""
    file_path = os.path.join(settings.MODEL_STORAGE_PATH, upload_file.filename)
    
    # Buat direktori jika belum ada
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)
    
    return file_path 