# Stock_Market_Forecasting_LSTM
# 📈 Stock Market Forecasting — LSTM + FastAPI (Prediction Dashboard)
A **FastAPI-powered AI web application** that predicts **stock market prices** for **BSE & NIFTY indices** using **LSTM neural networks**.  
This project combines **deep learning (TensorFlow/Keras)** with a **modern FastAPI dashboard**, creating a seamless stock forecasting experience 💹.

---

# 📈 Stock Market Forecasting — LSTM + FastAPI (Prediction Dashboard)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![LSTM](https://img.shields.io/badge/Model-LSTM-purple?logo=pytorch)](https://en.wikipedia.org/wiki/Long_short-term_memory)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellowgreen.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)](#)
[![Pandas](https://img.shields.io/badge/Data-Pandas-150458?logo=pandas)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?logo=scikit-learn)](https://scikit-learn.org/)

## 🌟 Features

### 📊 Predict Stock Prices
- Get **accurate forecasts** for **BSE** and **NIFTY** indices
- Customize prediction timeframe (**1-30 days**)
- View predictions in **clean table format**

### 🤖 LSTM Neural Network
- **Deep Learning** model trained on historical market data
- **10-day sliding window** for time series analysis
- **MinMaxScaler normalization** for optimal performance

### 🔐 Secure Authentication System
- **User registration** with email verification
- **Session-based authentication** (24-hour validity)
- **SHA-256 password hashing** for security
- **HTTP-only cookies** to prevent XSS attacks

### 🎨 Modern Web Interface
- **Responsive design** for all devices
- **Clean UI/UX** with intuitive navigation
- **Real-time predictions** without page reload
- **Historical data visualization**

### 📈 Multiple Market Indices
- **BSE (Bombay Stock Exchange)** predictions
- **NIFTY 50** index forecasting
- **Extensible architecture** for adding more indices

---

## 📸 Screenshots

### Login Page
<img width="721" height="915" alt="Image" src="https://github.com/user-attachments/assets/6786c250-2ea6-45f7-87a1-1b832d4db8f0" />

### Prediction Dashboard
<img width="1902" height="888" alt="Image" src="https://github.com/user-attachments/assets/e0ff33ed-e915-46e5-9d6e-bb3322a64725" />

### Forecast Results
<img width="1444" height="922" alt="Image" src="https://github.com/user-attachments/assets/968e24d3-a1dc-4d94-a731-bc79d7a05037" />

---

## 📋 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Model Details](#model-details)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Quick Start

### 📋 Prerequisites

Before you begin, ensure you have:
- ✅ **Python 3.8+** installed
- ✅ **pip** package manager
- ✅ **Git** for cloning the repository
- ✅ **Virtual environment** (recommended)

---

### ⚡ Installation Steps

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/stock-market-forecasting.git
cd stock-market-forecasting
```

#### 2️⃣ Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4️⃣ Verify Required Files

Make sure these files exist:
```
✓ app.py
✓ lstm_close_model.h5
✓ lstm_scaler.pkl
✓ synthetic_market.csv
✓ templates/ (folder with HTML files)
✓ static/ (folder for CSS/JS)
```

#### 5️⃣ Run the Application

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

#### 6️⃣ Open in Browser

Navigate to: **http://127.0.0.1:8000**

🎉 **You're all set!** Create an account and start forecasting!

---

## 📁 Project Structure

```
stock-market-forecasting/
│
├── app.py                          # Main FastAPI application
├── lstm_close_model.h5             # Trained LSTM model
├── lstm_scaler.pkl                 # Data scaler for normalization
├── synthetic_market.csv            # Historical market data
├── requirements.txt                # Python dependencies
│
├── templates/                      # HTML templates
│   ├── login.html                  # Login page
│   ├── register.html               # Registration page
│   ├── index.html                  # Home/forecast page
│   ├── BSE.html                    # BSE predictions
│   └── NIFTY.html                  # NIFTY predictions
│
├── static/                         # Static assets (CSS, JS)
│   └── styles.css
│
├── users.json                      # User credentials (auto-generated)
├── sessions.json                   # Active sessions (auto-generated)
│
└── training_scripts/               # Model training scripts
    ├── 01_generate_data.py
    └── 02_train_models.py
```

## 💻 How to Use

### 🔑 Step 1: Create Account
1. Navigate to **http://127.0.0.1:8000**
2. Click **"Register"**
3. Fill in:
   - Username
   - Email address
   - Password (min 6 characters)
   - Confirm password
4. Click **"Register"** button

### 🔓 Step 2: Login
1. Go to **Login** page
2. Enter your **username** and **password**
3. Session remains active for **24 hours**

### 📈 Step 3: Make Predictions

#### For BSE Index:
1. Click **"BSE"** in navigation
2. Enter **number of days** (1-30)
3. Click **"Forecast"**
4. View predicted closing prices

#### For NIFTY Index:
1. Click **"NIFTY"** in navigation
2. Enter **number of days** (1-30)
3. Click **"Forecast"**
4. View predicted closing prices

### 🚪 Step 4: Logout
- Click **"Logout"** button to end session
- Secure logout clears all session data

---

## 🔌 API Endpoints

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root redirect (to register or home) |
| GET | `/register` | Registration page |
| POST | `/register` | Create new user account |
| GET | `/login` | Login page |
| POST | `/login` | Authenticate user |
| GET | `/logout` | End user session |

### Forecasting

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/home` | Main forecast page | ✅ |
| POST | `/forecast` | Generate predictions | ✅ |
| GET | `/BSE` | BSE index page | ✅ |
| POST | `/forecast_bse` | BSE predictions | ✅ |
| GET | `/NIFTY` | NIFTY index page | ✅ |
| POST | `/forecast_nifty` | NIFTY predictions | ✅ |

## 🧠 LSTM Model Architecture

### 📐 Model Structure

```python
Input Layer (10 days of closing prices)
    ↓
LSTM Layer 1 (50 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 2 (50 units)
    ↓
Dropout (0.2)
    ↓
Dense Layer (25 units, ReLU)
    ↓
Output Layer (1 unit) → Predicted Price
```

### 🔢 Technical Specifications

| Parameter | Value |
|-----------|-------|
| **Model Type** | LSTM (Long Short-Term Memory) |
| **Framework** | TensorFlow 2.x / Keras |
| **Input Shape** | (10, 1) - 10-day window |
| **Output** | Single closing price prediction |
| **Optimizer** | Adam |
| **Loss Function** | Mean Squared Error (MSE) |
| **Scaler** | MinMaxScaler (0-1 range) |

### 📊 Training Details

- **Dataset**: Historical market data (synthetic/real)
- **Train-Test Split**: 80% training, 20% testing
- **Epochs**: 50 (with early stopping)
- **Batch Size**: 32
- **Validation Split**: 10% of training data

### 📈 Performance Metrics

```
Mean Absolute Error (MAE): [Your MAE value]
Root Mean Squared Error (RMSE): [Your RMSE value]
R² Score: [Your R² value]
Training Accuracy: [Your accuracy]
```

### 🔄 Prediction Process

1. **Input**: Last 10 days of closing prices
2. **Normalization**: Scale values to 0-1 range
3. **LSTM Processing**: Extract temporal patterns
4. **Prediction**: Generate next day's price
5. **Denormalization**: Convert back to actual price range
6. **Iteration**: Repeat for multi-day forecasts

---

## ⚙️ Configuration & Customization

### 🔧 Environment Variables (Optional)

Create a `.env` file in the root directory:

```env
# Server Configuration
HOST=127.0.0.1
PORT=8000
RELOAD=True

# Session Configuration
SESSION_EXPIRY_HOURS=24
SECRET_KEY=your-secret-key-here

# Model Configuration
MODEL_PATH=lstm_close_model.h5
SCALER_PATH=lstm_scaler.pkl
DATA_PATH=synthetic_market.csv

# Security Settings
MIN_PASSWORD_LENGTH=6
COOKIE_HTTPONLY=True
```

### 🛠️ Modifying Settings

Edit `app.py` to customize:

```python
# Session duration
SESSION_EXPIRY = timedelta(hours=24)

# Password requirements
MIN_PASSWORD_LENGTH = 6

# Forecast parameters
DEFAULT_FORECAST_DAYS = 5
MAX_FORECAST_DAYS = 30
LSTM_LOOKBACK_WINDOW = 10
```

### 📊 Using Real Market Data

Replace `synthetic_market.csv` with real data:

```python
# Required CSV format:
# date,open,high,low,close,volume
# 2024-01-01,100.5,105.2,99.8,103.4,1500000
# 2024-01-02,103.5,108.1,102.3,106.7,1600000
```

### 🔄 Retraining the Model

```bash
# Run the training script
python 02_train_models.py

# This will generate:
# - lstm_close_model.h5 (new model)
# - lstm_scaler.pkl (new scaler)
```

---

## 📦 Requirements

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
jinja2==3.1.2
python-multipart==0.0.6
pandas==2.1.3
numpy==1.24.3
tensorflow==2.15.0
scikit-learn==1.3.2
joblib==1.3.2
```

## 🔒 Security Features

- ✅ SHA-256 password hashing
- ✅ HTTP-only cookies
- ✅ Session expiry (24 hours)
- ✅ CSRF protection ready
- ✅ No plaintext password storage

## 🐛 Troubleshooting Guide

### ❌ Common Issues & Solutions

#### **Issue 1: "Module not found" Error**
```bash
# Solution:
pip install -r requirements.txt --upgrade
pip install tensorflow scikit-learn pandas fastapi
```

#### **Issue 2: Model File Not Loading**
```bash
# Error: "FileNotFoundError: lstm_close_model.h5"
# Solution: Ensure model files are in the root directory
ls lstm_close_model.h5 lstm_scaler.pkl  # Check files exist
```

#### **Issue 3: Template Not Found**
```bash
# Error: "TemplateNotFound: index.html"
# Solution: Verify templates folder structure
templates/
├── login.html
├── register.html
├── index.html
├── BSE.html
└── NIFTY.html
```

#### **Issue 4: Can't Access After Login**
```bash
# Solution: Clear browser cookies
# Chrome: Settings → Privacy → Clear browsing data → Cookies
# Firefox: Options → Privacy → Clear Data → Cookies
# Or use Incognito/Private mode
```

#### **Issue 5: TensorFlow Installation Error (Windows)**
```bash
# For Windows users with Python 3.11+:
pip install tensorflow-cpu  # CPU version (easier)
# OR
pip install tensorflow-gpu  # GPU version (requires CUDA)
```

#### **Issue 6: Port Already in Use**
```bash
# Error: "Address already in use"
# Solution: Change port or kill existing process
uvicorn app:app --reload --port 8001  # Use different port
# OR find and kill the process:
netstat -ano | findstr :8000  # Windows
lsof -i :8000  # Linux/Mac
```

#### **Issue 7: Prediction Returns NaN**
```bash
# Cause: Data normalization issue
# Solution: Retrain the model with proper data
python 02_train_models.py
```

### 📧 Still Having Issues?

Open an issue on GitHub with:
- ✅ Error message (full traceback)
- ✅ Python version (`python --version`)
- ✅ Operating system
- ✅ Steps to reproduce

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### 🔄 How to Contribute

#### 1️⃣ Fork the Repository
```bash
# Click "Fork" button on GitHub
git clone https://github.com/yourusername/stock-market-forecasting.git
cd stock-market-forecasting
```

#### 2️⃣ Create a Feature Branch
```bash
git checkout -b feature/AmazingFeature
```

#### 3️⃣ Make Your Changes
- Write clean, documented code
- Follow Python PEP 8 style guide
- Add comments for complex logic
- Update README if needed

#### 4️⃣ Test Your Changes
```bash
# Run the application
uvicorn app:app --reload

# Test all features:
# ✓ Registration
# ✓ Login
# ✓ BSE predictions
# ✓ NIFTY predictions
# ✓ Logout
```

#### 5️⃣ Commit & Push
```bash
git add .
git commit -m "✨ Add: Amazing new feature"
git push origin feature/AmazingFeature
```

#### 6️⃣ Open a Pull Request
- Go to your fork on GitHub
- Click **"New Pull Request"**
- Describe your changes
- Wait for review!

### 📝 Commit Message Guidelines

Use these prefixes:
- ✨ `Add:` New feature
- 🐛 `Fix:` Bug fix
- 📝 `Docs:` Documentation update
- ♻️ `Refactor:` Code refactoring
- 🎨 `Style:` Formatting changes
- ✅ `Test:` Adding tests
- ⚡ `Perf:` Performance improvement

Example:
```bash
git commit -m "✨ Add: Multi-stock symbol support"
git commit -m "🐛 Fix: Session timeout issue"
```

### 🎯 Areas for Contribution

- [ ] Add real-time data integration (Yahoo Finance API)
- [ ] Implement advanced charting (Chart.js/D3.js)
- [ ] Add email notifications
- [ ] Create portfolio tracking
- [ ] Add sentiment analysis
- [ ] Improve UI/UX design
- [ ] Write unit tests
- [ ] Add Docker support
- [ ] Create API documentation

---

## 🚀 Future Enhancements

### 🎯 Planned Features

#### 📊 **Advanced Analytics**
- [ ] Real-time stock data integration (Yahoo Finance API)
- [ ] Historical price comparison charts
- [ ] Technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Volume analysis and trends

#### 💹 **Multi-Stock Support**
- [ ] Support for individual stock symbols
- [ ] International market indices (S&P 500, FTSE, DAX)
- [ ] Cryptocurrency price predictions
- [ ] Commodity forecasting (Gold, Oil)

#### 📱 **Mobile & UI**
- [ ] Progressive Web App (PWA)
- [ ] Native mobile app (React Native)
- [ ] Dark mode toggle
- [ ] Interactive Chart.js/D3.js visualizations
- [ ] Real-time price updates (WebSockets)

#### 🔔 **Notifications & Alerts**
- [ ] Email notifications for price targets
- [ ] SMS alerts for significant predictions
- [ ] Custom alert thresholds
- [ ] Daily forecast summary emails

#### 🎓 **Advanced Features**
- [ ] Portfolio tracking and management
- [ ] Backtesting historical predictions
- [ ] Sentiment analysis from news/Twitter
- [ ] Multi-model ensemble predictions
- [ ] Risk assessment metrics
- [ ] Profit/Loss calculator

#### 🔧 **Developer Tools**
- [ ] REST API for predictions
- [ ] Docker containerization
- [ ] Kubernetes deployment
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Unit and integration tests
- [ ] API documentation (Swagger/OpenAPI)

#### 🌐 **Deployment**
- [ ] Deploy to Heroku
- [ ] Deploy to AWS EC2/Lambda
- [ ] Deploy to Google Cloud Run
- [ ] Deploy to Azure App Service

### 💡 Have an Idea?

Open an issue with the `enhancement` label and describe your feature request!

---

## 👨‍💻 Author

**Vibhav Raj**

[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/vibhav22-raj)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/vibhavraj/)
[![Email](https://img.shields.io/badge/Email-Contact-red?logo=gmail)](mailto:thevibhav2005@gmail.com)

---

## 🙏 Acknowledgments

Special thanks to:

- 🚀 **[FastAPI Team](https://fastapi.tiangolo.com/)** - For the amazing web framework
- 🧠 **[TensorFlow Team](https://www.tensorflow.org/)** - For the powerful ML library
- 📊 **[Pandas Community](https://pandas.pydata.org/)** - For data manipulation tools
- 🎨 **[Bootstrap](https://getbootstrap.com/)** - For UI components
- 💡 **Open Source Community** - For inspiration and support
- 📚 **Stock Market Researchers** - For financial modeling insights

### 📖 Resources & References

- [LSTM Networks Explained](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [TensorFlow Time Series Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Stock Market Prediction Research Papers](https://scholar.google.com/)

---

## 📞 Support & Contact

### 🐛 Found a Bug?
Open an issue on [GitHub Issues](https://github.com/yourusername/stock-market-forecasting/issues)

### 💬 Need Help?
- 📧 Email: thevibhav2005@gmail.com


### 💰 Support the Project
If you find this project helpful:

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Support-yellow?logo=buy-me-a-coffee)](https://buymeacoffee.com/yourusername)
[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub-pink?logo=github-sponsors)](https://github.com/sponsors/yourusername)

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### 📄 MIT License Summary:
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ❌ Liability
- ❌ Warranty

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/stock-market-forecasting?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/stock-market-forecasting?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/yourusername/stock-market-forecasting?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/stock-market-forecasting)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/stock-market-forecasting)
![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/stock-market-forecasting)

---

<div align="center">

### ⭐ Star this repository if you found it helpful!

**Made with ❤️ and Python**

🔮 *Predicting the future, one price at a time* 📈

</div>
