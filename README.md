# 📈 StockTrend AI — Stock Market Forecasting Model  

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
