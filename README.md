# 📈 StockTrend AI — Stock Market Forecasting Model  

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-API-red?logo=keras)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)



## 🧠 Overview  
**StockTrend AI** is a deep learning–based project designed to forecast stock market trends using historical data.  
It leverages **Long Short-Term Memory (LSTM)** networks — a specialized form of recurrent neural network (RNN) — to capture long-term dependencies in time-series stock data.  

This project demonstrates how AI can analyze and predict stock price movements by integrating techniques from **Machine Learning**, **Deep Learning**, and **Data Preprocessing**.

---

## 👨‍🏫 Mentor & Author  
**Mentor:** Palanivel Rajalingam  
**Developed by:** Vibhav Raj  

---

## ⚙️ AI Components Involved  

| Component | Description |
|------------|-------------|
| 🤖 **Machine Learning (ML)** | Learns from historical data for prediction tasks such as regression & classification. Forms the foundation for stock price forecasting. |
| 🧠 **Deep Learning (DL)** | Uses neural networks with multiple layers. LSTM (used here) captures time dependencies in stock price sequences. |
| 🗣️ **Natural Language Processing (NLP)** | Helps analyze financial news or tweets for sentiment analysis. (Not implemented but can enhance predictions.) |
| 👁️ **Computer Vision (CV)** | Processes images or charts (not used here but useful in fintech OCR tasks). |
| 🧬 **Generative AI (GenAI)** | Creates new data (like text or code). Not used, since our focus is **prediction**, not generation. |
| 🧠 **Agentic AI** | Autonomous decision-making AI agents (potential future extension). |

---

## 🧩 Implementation Steps  

### 1️⃣ Data Preprocessing  
- **Data Source:** Historical stock prices (e.g., from Yahoo Finance CSV).  
- **Feature Selection:** Used the `Close` price for forecasting.  
- **Missing Values:** Removed using `dropna()`.  
- **Scaling:** Applied `MinMaxScaler` to normalize data between 0–1 for stable LSTM training.

### 2️⃣ Train–Test Split  
- **Training Set:** 80%  
- **Testing Set:** 20%  
This ensures the model learns from historical data and is evaluated on unseen trends.

### 3️⃣ Model Training (LSTM)  
Built using **Keras Sequential API**  
- **Loss Function:** Mean Squared Error (MSE)  
- **Optimizer:** Adam  
- **Epochs:** 50–100 (tunable based on convergence)  

```python
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    LSTM(100, return_sequences=False),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')




⚙️ Installation & Running the Project
1️⃣ Clone the Repository
git clone https://github.com/vibhav22-raj/StockTrend_AI.git
cd StockTrend_AI

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate     # On Windows
# OR
source venv/bin/activate  # On macOS/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the App
uvicorn app:app --reload


Now open your browser and visit:

🌐 User Interface: http://127.0.0.1:8000

📊 Prediction Results: Generated after you upload or input data

🧠 How It Works Internally

1️⃣ Data Loading: Reads real or synthetic stock CSV files
2️⃣ Feature Engineering: Adds lag, rolling means, volatility
3️⃣ Model Training:

LSTM learns temporal patterns

Random Forest learns non-linear relationships
4️⃣ Prediction: Combines model outputs for final forecast
5️⃣ Visualization: Plots actual vs predicted prices

📦 Example Requirements
fastapi
uvicorn
tensorflow
scikit-learn
pandas
numpy
matplotlib
jinja2
python-multipart

🧰 Troubleshooting
Problem	Possible Solution
❌ ModuleNotFoundError	Run pip install -r requirements.txt
⚠️ Model not loading	Ensure .h5 and .pkl files are in project root
🔒 App not launching	Check Python v3.10+ and activate venv
🖼️ Static not loading	Ensure /static and /templates folders exist
🧑‍💻 Tech Stack
Layer	Technology
Frontend	HTML, CSS, JS, Bootstrap 5, Chart.js
Backend	FastAPI
Machine Learning	TensorFlow (LSTM), Scikit-Learn (Random Forest)
Data Handling	Pandas, NumPy
Visualization	Matplotlib, Chart.js
📸 Preview
🏠 User Interface

(Upload your stock dataset and view real-time forecasts)

📊 Result Page

(Shows trend prediction and model comparison)

🚀 Future Enhancements

🔄 Real-time stock price fetching via APIs

💾 Persistent storage using SQLite or PostgreSQL

🧭 Add ARIMA & Prophet for hybrid forecasting

🌙 Dark Mode dashboard UI

🤖 Integration with LM Studio for text-based insights

🧩 Credits

Developed with ❤️ by Vibhav Raj

Powered by FastAPI, TensorFlow, and LangChain AI Concepts

“Predicting tomorrow’s trends with today’s data 📊✨”
