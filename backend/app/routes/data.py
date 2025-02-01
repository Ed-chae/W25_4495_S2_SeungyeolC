import os
from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
from io import BytesIO
from app.services.data_ingestion import validate_excel

router = APIRouter()

@router.post("/upload-excel/")
async def upload_excel(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    try:
        # ✅ Debugging log
        print(f"Received file: {file.filename}")

        # ✅ Read file into memory
        file_data = await file.read()

        # ✅ Ensure file is not empty
        if not file_data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # ✅ Reset file pointer before passing to validation
        file_stream = BytesIO(file_data)
        file_stream.seek(0)  # Ensure file is read from the start

        # ✅ Validate Excel file
        df, errors = validate_excel(file_stream)

        # ✅ If validation fails, return errors
        if errors:
            print("Validation failed:", errors)
            return {"status": "error", "message": "Validation failed", "errors": errors}

        # ✅ Ensure the directory exists before saving the file
        save_path = "backend/data/cleaned_data.csv"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # ✅ Save cleaned CSV
        df.to_csv(save_path, index=False)

        return {"status": "success", "message": "File validated and saved"}

    except Exception as e:
        print(f"Unexpected server error: {str(e)}")  # Debugging
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
