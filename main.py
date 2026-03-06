from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

import joblib
import pandas as pd
import matplotlib.pyplot as plt

app = FastAPI()

model = joblib.load("model.pkl")

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, year: int = Form(...), month: int = Form(...), day: int = Form(...)):

    prediction = model.predict([[year, month, day]])
    rainfall = round(prediction[0],2)

    data = pd.read_csv("data/chennai_rainfall_data.csv")

    plt.figure()
    plt.plot(data["Rainfall_mm_day"])
    plt.title("Rainfall Trend")
    plt.xlabel("Days")
    plt.ylabel("Rainfall (mm)")

    plt.savefig("static/rainfall_plot.png")
    plt.close()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": rainfall
    })