# Cell 3: Load model and provide query functions
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # not used, but kept if you expand
from datetime import timedelta

MODEL_PATH = "rf_close_model.pkl"
SCALER_PATH = "scaler.pkl"
ARTIFACT_PATH = "model_artifacts.pkl"

# Load
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
artifacts = joblib.load(ARTIFACT_PATH)
feature_cols = artifacts["feature_columns"]

# Load the full raw and processed datasets
full_raw = pd.read_csv(artifacts["full_raw_csv"], parse_dates=["date"])
processed = pd.read_csv(artifacts["processed_features_csv"], parse_dates=["date"])

# Helper: create feature row given a historic df and last_date; returns df row for model
def build_feature_row(history_df, current_date):
    """
    history_df: full raw df with date, open, high, low, close, volume
    current_date: pd.Timestamp for which we will create features based on past data
    This function constructs the features matching 'feature_cols' for a row whose 'close' is known for current_date.
    """
    # we expect history_df to include current_date
    dfh = history_df.set_index("date").sort_index()
    if current_date not in dfh.index:
        raise ValueError("current_date must be present in history_df index for feature building.")
    # create a small helper series
    row = {}
    # lags: lag_1 = close at day-1 etc
    for col in feature_cols:
        if col.startswith("lag_"):
            lag = int(col.split("_")[1])
            lag_date = current_date - pd.Timedelta(days=lag)
            row[col] = dfh.at[lag_date, "close"] if lag_date in dfh.index else np.nan
        elif col.startswith("roll_mean_"):
            w = int(col.split("_")[-1])
            window_end = current_date - pd.Timedelta(days=1)
            window_start = window_end - pd.Timedelta(days=w-1)
            window = dfh.loc[window_start:window_end]["close"]
            row[col] = window.mean() if not window.empty else np.nan
        elif col.startswith("roll_std_"):
            w = int(col.split("_")[-1])
            window_end = current_date - pd.Timedelta(days=1)
            window_start = window_end - pd.Timedelta(days=w-1)
            window = dfh.loc[window_start:window_end]["close"]
            row[col] = window.std() if not window.empty else np.nan
        elif col == "pct_change_1":
            prev = current_date - pd.Timedelta(days=1)
            if prev in dfh.index and (dfh.at[prev, "close"] != 0):
                row[col] = (dfh.at[prev, "close"] - dfh.at[prev - pd.Timedelta(days=1), "close"]) / dfh.at[prev - pd.Timedelta(days=1), "close"] if (prev - pd.Timedelta(days=1)) in dfh.index else 0.0
            else:
                row[col] = 0.0
        elif col == "vol_ma_7":
            window_end = current_date - pd.Timedelta(days=1)
            window_start = window_end - pd.Timedelta(days=6)
            window = dfh.loc[window_start:window_end]["volume"]
            row[col] = window.mean() if not window.empty else dfh.at[window_end, "volume"] if window_end in dfh.index else np.nan
        else:
            # fallback: try to pull that column if exists in raw
            if col in dfh.columns:
                row[col] = dfh.at[current_date, col]
            else:
                row[col] = np.nan
    # convert to df
    feat_df = pd.DataFrame([row])
    return feat_df

# Iterative forecasting function: forecast next n days using last available day in full_raw
# ==========================
# Updated Iterative Forecast Function
# ==========================

def forecast_next_days(n=5):
    """
    Iteratively forecast next n days of close price.
    Uses the last row in full_raw as the most recent observed day.
    """
    hist = full_raw.copy().set_index("date").sort_index()
    last_date = hist.index.max()
    results = []
    history = hist.copy()

    for step in range(1, n + 1):
        target_date = last_date + pd.Timedelta(days=step)
        current_date_for_features = target_date - pd.Timedelta(days=1)

        # If current_date_for_features not in history, create synthetic row
        if current_date_for_features not in history.index:
            last_row = history.iloc[-1]
            synth = last_row.copy()
            synth.name = current_date_for_features
            # ✅ Use pd.concat instead of append
            history = pd.concat([history, pd.DataFrame([synth.values], index=[current_date_for_features], columns=history.columns)])

        # Build features referencing history
        feat_row = build_feature_row(history.reset_index().rename(columns={"index": "date"}), current_date_for_features)

        # Scale features
        X = scaler.transform(feat_row[feature_cols].ffill().bfill())
        pred_close = model.predict(X)[0]

        results.append({"date": target_date.strftime("%Y-%m-%d"), "pred_close": float(pred_close)})

        # Append predicted day to history for next iteration
        avg_vol = int(history["volume"].tail(7).mean())
        new_row = {
            "open": pred_close,
            "high": pred_close * 1.002,
            "low": pred_close * 0.998,
            "close": pred_close,
            "volume": avg_vol
        }
        # ✅ Use pd.concat instead of append
        history = pd.concat([history, pd.DataFrame([new_row], index=[target_date])])

    return pd.DataFrame(results)


# ==========================
# Updated Predict Close for Specific Date
# ==========================

def predict_date(date_str):
    """
    Predict close for a specific future date.
    If the date is beyond the last observed, extend history iteratively.
    """
    date = pd.to_datetime(date_str)
    history = full_raw.copy().set_index("date").sort_index()
    last_date = history.index.max()
    current_date = date - pd.Timedelta(days=1)

    # If current_date is beyond last observed, extend history iteratively
    if current_date > last_date:
        steps = (current_date - last_date).days
        _ = forecast_next_days(steps)  # this internally extends full_raw for next steps

        # Rebuild local history to include synthetic rows
        history = full_raw.copy().set_index("date").sort_index()
        last_date = history.index.max()
        while current_date > last_date:
            last_row = history.iloc[-1]
            synth = last_row.copy()
            synth.name = last_date + pd.Timedelta(days=1)
            history = pd.concat([history, pd.DataFrame([synth.values], index=[synth.name], columns=history.columns)])
            last_date = history.index.max()

    # Now build features safely
    feat_row = build_feature_row(history.reset_index().rename(columns={"index": "date"}), current_date)
    X = scaler.transform(feat_row[feature_cols].ffill().bfill())
    pred_close = model.predict(X)[0]

    return {"date": date_str, "pred_close": float(pred_close)}



# ==========================
# Demo Usage
# ==========================
print("Forecast next 5 days (iterative):")
print(forecast_next_days(5))

print("\nPredict close on a date (example):")
print(predict_date((full_raw['date'].max() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")))
