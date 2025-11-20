from fastapi import FastAPI
import pandas as pd
import numpy as np
from scipy.signal import medfilt, savgol_filter
from datetime import datetime

app = FastAPI()

@app.post("/clean")
async def clean_data(payload: dict):

    # STEP 1 — Extract data
    records = payload.get("data", [])
    if not records:
        return {"error": "No data received"}

    df = pd.DataFrame(records)

    # Ensure the required fields exist
    if "_time" not in df.columns or "_value" not in df.columns:
        return {"error": "Missing '_time' or '_value' fields"}

    # STEP 2 — Convert time
    df["_time"] = pd.to_datetime(df["_time"], errors="coerce")

    df = df.dropna(subset=["_time"])
    df = df.sort_values("_time")

    # STEP 3 — Convert values
    values = df["_value"].astype(float).values

    # Remove impossible sensor readings
    values = np.where((values < 0) | (values > 0.60), np.nan, values)

    # ========== SAFE MEDIAN FILTER ==========
    if len(values) >= 5:
        med = medfilt(values, kernel_size=5)
    else:
        med = values.copy()

    # Step 5 — Spike detection
    diff = np.abs(values - med)
    cleaned = np.where(diff > 0.20, med, values)

    # ========== SAFE SAVITZKY-GOLAY ==========
    if len(cleaned) >= 7:
        # window must be odd and <= len(cleaned)
        win = len(cleaned) if len(cleaned) % 2 == 1 else len(cleaned) - 1
        win = max(win, 7)  # ensure minimum window of 7
        smooth = savgol_filter(cleaned, window_length=win, polyorder=2)
    else:
        smooth = cleaned.copy()

    df["cleaned"] = smooth

    # Convert timestamps to ISO strings for JSON
    df["_time"] = df["_time"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    return df[["_time", "cleaned"]].to_dict(orient="records")
