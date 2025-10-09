from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from datetime import datetime
import random

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
def wel_come(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict():
    # Example: generate a random stock price
    predicted_price = round(random.uniform(100, 500), 2)
    return JSONResponse(content=[{
        "date": datetime.now().strftime("%Y-%m-%d"),
        "pred_close": predicted_price
    }])
