from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import uuid

from preprocessing import basic_eda, handle_missing, split_data
from train import train
from utils import save_run

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = "data/cleaned.csv"

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    df = handle_missing(df, "drop")
    df.to_csv(DATA_PATH, index=False)
    return basic_eda(df)

@app.post("/train")
async def train_model(payload: dict):
    df = pd.read_csv(DATA_PATH)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_val, y_train, y_val = split_data(X, y)

    history, diagnostics = train(
        payload["model_type"],
        X_train,
        X_val,
        y_train,
        y_val,
        payload
    )

    run_id = str(uuid.uuid4())
    save_run(run_id, {
        "payload": payload,
        "history": history,
        "diagnostics": diagnostics
    })

    return {
        "history": history,
        "diagnostics": diagnostics
    }
