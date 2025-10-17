from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

app = FastAPI(title="Market Forecasting Simulator")

# IMPORTANT: Ensure your templates folder is correctly named "templates"
templates = Jinja2Templates(directory="templates")

# Mount static files (optional, since we're using Tailwind CDN for main styling)
# Keep this if you need to serve other assets like images or custom CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Data Loading (Placeholder for stock_data.csv) ---
STOCK_DATA_PATH = "stock_data.csv"
try:
    if os.path.exists(STOCK_DATA_PATH):
        df = pd.read_csv(STOCK_DATA_PATH)
    else:
        # Create an empty DataFrame if the file is missing to avoid crashes
        df = pd.DataFrame(columns=['date', 'close'])
        print(f"⚠️ Warning: {STOCK_DATA_PATH} not found. Using simulated data.")
except Exception as e:
    print(f"Error loading {STOCK_DATA_PATH}: {e}")
    df = pd.DataFrame(columns=['date', 'close'])


# --- Utility Function for Simulated Forecast ---
def generate_simulated_forecast(days: int, min_val: float, max_val: float):
    """Generates random forecast data for the next 'days'."""
    forecast = []
    start_date = datetime.now()
    
    for i in range(days):
        forecast_date = start_date + timedelta(days=i+1)
        predicted_close = round(random.uniform(min_val, max_val), 2)
        forecast.append({
            "date": forecast_date.strftime("%Y-%m-%d"),
            "pred_close": predicted_close
        })
    return forecast

# --- Home Route ---
@app.get("/", response_class=HTMLResponse)
async def wel_come(request: Request):
    """Renders the main index page."""
    # Pass an empty forecast array initially to prevent Jinja errors
    return templates.TemplateResponse("index.html", {"request": request, "forecast": []})

# --- BSE Routes ---

@app.get("/BSE", response_class=HTMLResponse)
async def bse_page(request: Request):
    """Renders the BSE page (GET request)."""
    return templates.TemplateResponse("bse.html", {"request": request, "forecast": []})

@app.post("/forecast_bse", response_class=HTMLResponse)
async def forecast_bse(request: Request, days: int = Form(...)):
    """Handles the BSE forecast form submission (POST request)."""
    forecast = generate_simulated_forecast(days, min_val=100.0, max_val=500.0)
    
    return templates.TemplateResponse(
        "bse.html",
        {"request": request, "forecast": forecast}
    )

# --- NIFTY Routes ---

@app.get("/NIFTY", response_class=HTMLResponse)
async def nifty_page(request: Request):
    """Renders the NIFTY page (GET request)."""
    return templates.TemplateResponse("nifty.html", {"request": request, "forecast": []})

@app.post("/forecast_nifty", response_class=HTMLResponse)
async def forecast_nifty(request: Request, days: int = Form(...)):
    """Handles the NIFTY forecast form submission (POST request)."""
    # Different range for NIFTY simulation
    forecast = generate_simulated_forecast(days, min_val=15000.0, max_val=25000.0) 
    
    return templates.TemplateResponse(
        "nifty.html",
        {"request": request, "forecast": forecast}
    )

# --- Simple JSON Prediction Route ---
@app.post("/predict", response_class=JSONResponse)
async def predict():
    """Returns a single random prediction as JSON."""
    predicted_price = round(random.uniform(100, 500), 2)
    return [{
        "date": datetime.now().strftime("%Y-%m-%d"),
        "pred_close": predicted_price
    }]
