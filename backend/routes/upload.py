from fastapi import APIRouter, File, UploadFile, HTTPException
import requests

router = APIRouter()

# ✅ Redirect File Uploads to Flask API
@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        files = {"file": (file.filename, await file.read(), file.content_type)}
        response = requests.post("http://127.0.0.1:5000/upload", files=files)

        # ✅ Check if Flask API returned an error
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Flask API Error: {response.text}")

        return response.json()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
