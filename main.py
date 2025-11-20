from fastapi import FastAPI
import pandas as pd
import numpy as np
from scipy.signal import medfilt, savgol_filter

app = FastAPI()

@app.post("/clean")
async def clean_data(payload: dict):
    records = payload["data"]
    df = pd.DataFrame(records)

    df["_time"] = pd.to_datetime(df["_time"])
    df = df.sort_values("_time")

    values = df["_value"].astype(float).values

    # Remove impossible values
    values = np.where((values < 0) | (values > 0.60), np.nan, values)

    # Rolling median
    med = medfilt(values, kernel_size=5)

    # Spike detection
    diff = np.abs(values - med)
    cleaned = np.where(diff > 0.20, med, values)

    # Smooth curve
    smooth = savgol_filter(cleaned, window_length=7, polyorder=2)

    df["cleaned"] = smooth
    return df[["_time", "cleaned"]].to_dict(orient="records")
