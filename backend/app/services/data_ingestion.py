import os
import pandas as pd
from fastapi import HTTPException
from io import BytesIO

# Required columns and their expected types
REQUIRED_COLUMNS = {
    "CustomerID": "int64",
    "OrderDate": "datetime64[ns]",
    "Revenue": "float64",
    "Feedback": "string"
}

def validate_excel(file: BytesIO):
    """
    Reads and validates an uploaded Excel file.
    - Ensures required columns exist.
    - Converts data types correctly.
    - Handles missing values by dropping rows.
    - Catches and logs specific errors.
    """
    try:
        # ✅ Explicitly use 'openpyxl' for .xlsx files
        try:
            df = pd.read_excel(file, engine="openpyxl")
        except ImportError:
            raise HTTPException(status_code=400, detail="Missing 'openpyxl'. Install it using: pip install openpyxl.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading Excel file: {str(e)}")

        errors = []

        # ✅ Check if all required columns exist
        missing_columns = [col for col in REQUIRED_COLUMNS.keys() if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")

        if errors:
            return None, errors  # Stop processing if columns are missing

        # ✅ Convert data types and handle errors
        for col, dtype in REQUIRED_COLUMNS.items():
            try:
                if dtype == "datetime64[ns]":
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    if df[col].isna().sum() > 0:
                        errors.append(f"Invalid date format detected in column '{col}'")
                else:
                    df[col] = df[col].astype(dtype)
            except Exception as e:
                errors.append(f"Error converting {col} to {dtype}: {str(e)}")

        # ✅ Drop rows with missing values
        df = df.dropna()

        return df, errors

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Error parsing Excel file. Ensure it's a valid format.")
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File processing error: {str(e)}")

def save_cleaned_data(df: pd.DataFrame):
    """
    Saves the cleaned DataFrame to a CSV file inside 'backend/data/'.
    Ensures the directory exists before saving.
    """
    save_path = "backend/data/cleaned_data.csv"
    
    try:
        # ✅ Ensure directory exists before saving
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # ✅ Save cleaned data
        df.to_csv(save_path, index=False)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

def load_cleaned_data():
    """
    Loads the cleaned data from the saved CSV file.
    Returns a DataFrame or None if the file doesn't exist.
    """
    file_path = "backend/data/cleaned_data.csv"
    
    if not os.path.exists(file_path):
        return None

    try:
        df = pd.read_csv(file_path, parse_dates=["OrderDate"])
        return df

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading cleaned data: {str(e)}")
