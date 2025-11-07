from fastapi import FastAPI, Request, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import json
import os
from datetime import datetime, timedelta
from typing import Optional
import secrets
import hashlib

# -------------------------------------------------
# 🌐 Initialize FastAPI
# -------------------------------------------------
app = FastAPI(title="Stock Market Forecasting (LSTM)")

# Serve static files (for CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load HTML templates
templates = Jinja2Templates(directory="templates")

# -------------------------------------------------
# 🔐 Authentication Setup
# -------------------------------------------------
USERS_FILE = "users.json"
SESSIONS_FILE = "sessions.json"

def load_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def load_sessions():
    """Load sessions from JSON file"""
    if os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_sessions(sessions):
    """Save sessions to JSON file"""
    with open(SESSIONS_FILE, 'w') as f:
        json.dump(sessions, f, indent=4)

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password"""
    return hash_password(plain_password) == hashed_password

def create_session(username: str) -> str:
    """Create a new session for user"""
    sessions = load_sessions()
    session_token = secrets.token_urlsafe(32)
    expiry = (datetime.now() + timedelta(hours=24)).isoformat()
    
    sessions[session_token] = {
        "username": username,
        "expiry": expiry
    }
    save_sessions(sessions)
    return session_token

def get_current_user(request: Request) -> Optional[str]:
    """Get current logged-in user from session"""
    session_token = request.cookies.get("session_token")
    if not session_token:
        return None
    
    sessions = load_sessions()
    session = sessions.get(session_token)
    
    if not session:
        return None
    
    # Check if session expired
    expiry = datetime.fromisoformat(session["expiry"])
    if datetime.now() > expiry:
        del sessions[session_token]
        save_sessions(sessions)
        return None
    
    return session["username"]

def require_auth(request: Request):
    """Dependency to require authentication"""
    username = get_current_user(request)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": "/login"}
        )
    return username

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
# 🔐 AUTHENTICATION ROUTES
# -------------------------------------------------

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    """Login Page"""
    username = get_current_user(request)
    if username:
        return RedirectResponse(url="/home", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@app.post("/login", response_class=HTMLResponse)
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle login"""
    users = load_users()
    
    if username not in users or not verify_password(password, users[username]["password"]):
        return templates.TemplateResponse("login.html", {
            "request": request, 
            "error": "Invalid username or password"
        })
    
    session_token = create_session(username)
    response = RedirectResponse(url="/home", status_code=303)
    response.set_cookie(key="session_token", value=session_token, httponly=True, max_age=86400)
    return response

@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    """Registration Page"""
    username = get_current_user(request)
    if username:
        return RedirectResponse(url="/home", status_code=303)
    return templates.TemplateResponse("register.html", {"request": request, "error": None, "success": None})

@app.post("/register", response_class=HTMLResponse)
def register(
    request: Request, 
    username: str = Form(...), 
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...)
):
    """Handle registration"""
    users = load_users()
    
    if username in users:
        return templates.TemplateResponse("register.html", {
            "request": request, 
            "error": "Username already exists",
            "success": None
        })
    if len(password) < 6:
        return templates.TemplateResponse("register.html", {
            "request": request, 
            "error": "Password must be at least 6 characters",
            "success": None
        })
    if password != confirm_password:
        return templates.TemplateResponse("register.html", {
            "request": request, 
            "error": "Passwords do not match",
            "success": None
        })
    
    users[username] = {
        "email": email,
        "password": hash_password(password),
        "created_at": datetime.now().isoformat()
    }
    save_users(users)
    
    return templates.TemplateResponse("register.html", {
        "request": request, 
        "error": None,
        "success": "Registration successful! Please login."
    })

@app.get("/logout")
def logout(request: Request):
    """Handle logout"""
    session_token = request.cookies.get("session_token")
    if session_token:
        sessions = load_sessions()
        if session_token in sessions:
            del sessions[session_token]
            save_sessions(sessions)
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("session_token")
    return response

# -------------------------------------------------
# 🏠 REDIRECT & HOME ROUTES
# -------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def root_redirect(request: Request):
    """Redirect to register or home depending on session"""
    username = get_current_user(request)
    if username:
        return RedirectResponse(url="/home", status_code=303)
    return RedirectResponse(url="/register", status_code=303)

@app.get("/home", response_class=HTMLResponse)
def home(request: Request, username: str = Depends(require_auth)):
    """Home Page (Protected)"""
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "forecast": None,
        "username": username
    })

@app.post("/forecast", response_class=HTMLResponse)
def forecast(request: Request, days: int = Form(...), username: str = Depends(require_auth)):
    """Forecast route for homepage (Protected)"""
    forecasted = forecast_next_days_lstm(df, n_days=days)
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "forecast": forecasted,
        "username": username
    })

# -------------------------------------------------
# 💹 BSE Routes (Protected)
# -------------------------------------------------

@app.get("/BSE", response_class=HTMLResponse)
def bse_page(request: Request, username: str = Depends(require_auth)):
    """BSE Forecast Page (Protected)"""
    return templates.TemplateResponse("BSE.html", {
        "request": request, 
        "forecast": None,
        "username": username
    })

@app.post("/forecast_bse", response_class=HTMLResponse)
def forecast_bse(request: Request, days: int = Form(...), username: str = Depends(require_auth)):
    """Forecast for BSE (Protected)"""
    forecasted = forecast_next_days_lstm(df, n_days=days)
    return templates.TemplateResponse("BSE.html", {
        "request": request, 
        "forecast": forecasted,
        "username": username
    })

# -------------------------------------------------
# 📊 NIFTY Routes (Protected)
# -------------------------------------------------

@app.get("/NIFTY", response_class=HTMLResponse)
def nifty_page(request: Request, username: str = Depends(require_auth)):
    """NIFTY Forecast Page (Protected)"""
    return templates.TemplateResponse("NIFTY.html", {
        "request": request, 
        "forecast": None,
        "username": username
    })

@app.post("/forecast_nifty", response_class=HTMLResponse)
def forecast_nifty(request: Request, days: int = Form(...), username: str = Depends(require_auth)):
    """Forecast for NIFTY (Protected)"""
    forecasted = forecast_next_days_lstm(df, n_days=days)
    return templates.TemplateResponse("NIFTY.html", {
        "request": request, 
        "forecast": forecasted,
        "username": username
    })
