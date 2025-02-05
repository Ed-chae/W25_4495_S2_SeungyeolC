from fastapi import APIRouter, File, UploadFile
from file_processing import process_excel

router = APIRouter()

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    return process_excel(contents)
