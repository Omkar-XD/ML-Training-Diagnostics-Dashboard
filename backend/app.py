
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import uuid
import os

from preprocessing import basic_eda, handle_missing, split_data
from train import train
from utils import save_run

app = FastAPI()

# Allow frontend domains
origins = [
    "http://localhost:5173",
    "https://ml-training-diagnostics-dashboard.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

DATA_PATH = "data/cleaned.csv"

@app.get("/")
def root():
    return {"message": "ML Training Diagnostics API running"}

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV is empty"
            )

        # handle missing values
        df = handle_missing(df, "drop")

        # save cleaned dataset
        df.to_csv(DATA_PATH, index=False)

        eda_result = basic_eda(df)

        return {
            "message": "File uploaded successfully",
            "rows": int(len(df)),
            "columns": list(df.columns),
            "eda": eda_result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/train")
async def train_model(payload: dict):
    if not os.path.exists(DATA_PATH):
        raise HTTPException(
            status_code=400,
            detail="No dataset uploaded. Please upload CSV first."
        )

    df = pd.read_csv(DATA_PATH)

    if df.shape[1] < 2:
        raise HTTPException(
            status_code=400,
            detail="Dataset must contain features and a target column"
        )

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_val, y_train, y_val = split_data(X, y)

    model_type = payload.get("model_type")

    if not model_type:
        raise HTTPException(
            status_code=400,
            detail="model_type is required"
        )

    history, diagnostics = train(
        model_type,
        X_train,
        X_val,
        y_train,
        y_val,
        payload
    )

    run_id = str(uuid.uuid4())

    save_run(
        run_id,
        {
            "payload": payload,
            "history": history,
            "diagnostics": diagnostics
        }
    )

    return {
        "run_id": run_id,
        "history": history,
        "diagnostics": diagnostics
    }

