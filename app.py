from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -------------------------------------------------
# 🌐 Initialize FastAPI
# -------------------------------------------------
app = FastAPI(title="Stock Market Forecasting (LSTM)")

# Serve static files (for CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load HTML templates
templates = Jinja2Templates(directory="templates")

# -------------------------------------------------
# 🧠 Load model, scaler, and data
# -------------------------------------------------
try:
    lstm_model = load_model("lstm_close_model.h5", compile=False)
    scaler = joblib.load("lstm_scaler.pkl")
    df = pd.read_csv("synthetic_market.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
except Exception as e:
    print(f"⚠️ Error loading model or data: {e}")
    df = pd.DataFrame(columns=["date", "close"])

# -------------------------------------------------
# 📈 Forecast Function
# -------------------------------------------------
def forecast_next_days_lstm(df, n_days=5, n_steps=10):
    """
    Forecast next N days using trained LSTM model.
    """
    if df.empty:
        return []

    close_prices = df["close"].values
    forecasted = []
    last_sequence = close_prices[-n_steps:]

    for day in range(1, n_days + 1):
        seq_scaled = scaler.transform(last_sequence.reshape(-1, 1)).reshape(1, n_steps, 1)
        pred_scaled = lstm_model.predict(seq_scaled, verbose=0)
        pred_close = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]

        forecast_date = df["date"].max() + pd.Timedelta(days=day)
        forecasted.append({
            "date": forecast_date.strftime("%Y-%m-%d"),
            "pred_close": round(float(pred_close), 2)
        })

        # Update the sequence
        last_sequence = np.append(last_sequence[1:], pred_close)

    return forecasted

# -------------------------------------------------
# 🏠 ROUTES
# -------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Home Page"""
    return templates.TemplateResponse("index.html", {"request": request, "forecast": None})

@app.post("/forecast", response_class=HTMLResponse)
def forecast(request: Request, days: int = Form(...)):
    """Forecast route for homepage"""
    forecasted = forecast_next_days_lstm(df, n_days=days)
    return templates.TemplateResponse("index.html", {"request": request, "forecast": forecasted})

# -------------------------------------------------
# 💹 BSE Routes
# -------------------------------------------------

@app.get("/BSE", response_class=HTMLResponse)
def bse_page(request: Request):
    """BSE Forecast Page"""
    return templates.TemplateResponse("BSE.html", {"request": request, "forecast": None})

@app.post("/forecast_bse", response_class=HTMLResponse)
def forecast_bse(request: Request, days: int = Form(...)):
    """Forecast for BSE"""
    forecasted = forecast_next_days_lstm(df, n_days=days)
    return templates.TemplateResponse("BSE.html", {"request": request, "forecast": forecasted})

# -------------------------------------------------
# 📊 NIFTY Routes
# -------------------------------------------------

@app.get("/NIFTY", response_class=HTMLResponse)
def nifty_page(request: Request):
    """NIFTY Forecast Page"""
    return templates.TemplateResponse("NIFTY.html", {"request": request, "forecast": None})

@app.post("/forecast_nifty", response_class=HTMLResponse)
def forecast_nifty(request: Request, days: int = Form(...)):
    """Forecast for NIFTY"""
    forecasted = forecast_next_days_lstm(df, n_days=days)
    return templates.TemplateResponse("NIFTY.html", {"request": request, "forecast": forecasted})
