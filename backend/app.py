from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import uuid
import os

from preprocessing import basic_eda, handle_missing, split_data
from train import train
from utils import save_run

app = FastAPI()

# =========================
# CORS CONFIG
# =========================

default_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://ml-training-diagnostics-dashboard.vercel.app",
]

env_origins = os.getenv("ALLOWED_ORIGINS", "")

if env_origins:
    origins = [o.strip() for o in env_origins.split(",") if o.strip()]
else:
    origins = default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\\.vercel\\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# STORAGE SETUP
# =========================

os.makedirs("data", exist_ok=True)

DATA_PATH = "data/cleaned.csv"

# =========================
# ROOT ROUTE
# =========================

@app.get("/")
def root():
    return {
        "message": "ML Training Diagnostics API running"
    }

# =========================
# CSV UPLOAD
# =========================

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):

    try:

        # Read CSV
        df = pd.read_csv(file.file)

        # Validate dataset
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV is empty"
            )

        # Handle missing values
        df = handle_missing(df, "fill")

        # Save cleaned dataset
        df.to_csv(DATA_PATH, index=False)

        # Basic EDA
        eda_result = basic_eda(df)

        return {
            "message": "File uploaded successfully",
            "rows": int(len(df)),
            "columns": list(df.columns),
            "shape": list(df.shape),
            "missing_values": df.isnull().sum().to_dict(),
            "eda": eda_result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# =========================
# MODEL TRAINING
# =========================

@app.post("/train")
async def train_model(payload: dict):

    # Ensure dataset exists
    if not os.path.exists(DATA_PATH):

        raise HTTPException(
            status_code=400,
            detail="No dataset uploaded. Please upload CSV first."
        )

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Validate dataset structure
    if df.shape[1] < 2:

        raise HTTPException(
            status_code=400,
            detail="Dataset must contain at least one feature and one target column"
        )

    # =========================
    # TARGET COLUMN
    # =========================

    target_column = payload.get("target_column")

    if not target_column:

        raise HTTPException(
            status_code=400,
            detail="target_column is required"
        )

    if target_column not in df.columns:

        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' not found in dataset"
        )

    # =========================
    # SPLIT FEATURES / TARGET
    # =========================

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # =========================
    # PREPROCESS + SPLIT
    # =========================

    X_train, X_val, y_train, y_val = split_data(X, y)

    # =========================
    # MODEL TYPE
    # =========================

    model_type = payload.get("model_type")

    if not model_type:

        raise HTTPException(
            status_code=400,
            detail="model_type is required"
        )

    # =========================
    # TRAIN MODEL
    # =========================

    history, diagnostics = train(
        model_type,
        X_train,
        X_val,
        y_train,
        y_val,
        payload
    )

    # =========================
    # SAVE RUN
    # =========================

    run_id = str(uuid.uuid4())

    save_run(
        run_id,
        {
            "payload": payload,
            "history": history,
            "diagnostics": diagnostics
        }
    )

    # =========================
    # RESPONSE
    # =========================

    return {
        "run_id": run_id,
        "history": history,
        "diagnostics": diagnostics,
        "features_used": list(X_train.columns),
        "target_column": target_column
    }