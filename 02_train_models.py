# Cell 2: Train a RandomForestRegressor model and save model + artifacts
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Import numpy for square root
import numpy as np 
# The original code was missing 'import pandas as pd' but it is implied by usage
import pandas as pd

# Paths to save
MODEL_PATH = "rf_close_model.pkl"
SCALER_PATH = "scaler.pkl"
ARTIFACT_PATH = "model_artifacts.pkl"  # will store feature list and last known data for forecasting

# Load processed train/test
train_df = pd.read_csv("market_train.csv")
test_df = pd.read_csv("market_test.csv")

# Define features
feature_cols = [c for c in train_df.columns if c not in ["date", "target_close"]]

print("Features used:", feature_cols)

# Scale numeric features (volume might be large)
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols])
y_train = train_df["target_close"].values
X_test = scaler.transform(test_df[feature_cols])
y_test = test_df["target_close"].values

# Train model
model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

# --- CORRECTED LINE FOR RMSE CALCULATION ---
# Calculate MSE first, then take the square root to get RMSE.
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse) 
# --- END CORRECTION ---

print(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# Save artifacts
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
# Save feature list and full original df (to know last observed window for iterative forecast)
artifacts = {
    "feature_columns": feature_cols,
    "full_raw_csv": "synthetic_market.csv",  # file we generated earlier
    "processed_features_csv": "synthetic_market_features.csv"
}
joblib.dump(artifacts, ARTIFACT_PATH)

print(f"Saved model to {MODEL_PATH}, scaler to {SCALER_PATH}, artifacts to {ARTIFACT_PATH}")

# ==========================
# Train RandomForest and Evaluate
# ==========================
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np # <-- ADDED: Import numpy for square root

# Paths
MODEL_PATH = "rf_close_model.pkl"
SCALER_PATH = "scaler.pkl"
ARTIFACT_PATH = "model_artifacts.pkl"

# Load processed train/test
train_df = pd.read_csv("market_train.csv")
test_df = pd.read_csv("market_test.csv")

# Define features
feature_cols = [c for c in train_df.columns if c not in ["date", "target_close"]]
print("Features used:", feature_cols)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols])
y_train = train_df["target_close"].values
X_test = scaler.transform(test_df[feature_cols])
y_test = test_df["target_close"].values

# Train model
model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

# --- CORRECTED LINES FOR RMSE CALCULATION ---
# 1. Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, preds) 
# 2. Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse) 
# --- END CORRECTION ---

r2 = r2_score(y_test, preds)
print(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# ==========================
# Plot Predictions vs Actuals
# ==========================
plt.figure(figsize=(12,5))
# The 'date' column contains strings and needs to be converted to datetime objects for proper plotting.
# This assumes 'date' is a column in test_df.
dates = pd.to_datetime(test_df["date"])
plt.plot(dates, y_test, label="Actual Close", color="blue")
plt.plot(dates, preds, label="Predicted Close", color="red", alpha=0.7)
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("RandomForest Regressor Predictions vs Actuals")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# ==========================
# Save model and artifacts
# ==========================
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
artifacts = {
    "feature_columns": feature_cols,
    "full_raw_csv": "synthetic_market.csv",  
    "processed_features_csv": "synthetic_market_features.csv"
}
joblib.dump(artifacts, ARTIFACT_PATH)

print(f"✅ Saved model to {MODEL_PATH}, scaler to {SCALER_PATH}, artifacts to {ARTIFACT_PATH}")

# ==========================
# 1. Load dataset
# ==========================
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Load synthetic market data
df = pd.read_csv("synthetic_market.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# ==========================
# 2. Feature Engineering
# ==========================
def create_time_series_features(df, target="close"):
    df_feat = df.copy()
    # Lags
    for lag in range(1, 4):
        df_feat[f"lag_{lag}"] = df_feat[target].shift(lag)
    # Rolling statistics
    df_feat["roll_mean_3"] = df_feat[target].shift(1).rolling(window=3).mean()
    df_feat["roll_mean_7"] = df_feat[target].shift(1).rolling(window=7).mean()
    df_feat["roll_std_3"] = df_feat[target].shift(1).rolling(window=3).std()
    df_feat["roll_std_7"] = df_feat[target].shift(1).rolling(window=7).std()
    # Percent change
    df_feat["pct_change_1"] = df_feat[target].pct_change(1)
    df_feat["pct_change_3"] = df_feat[target].pct_change(3)
    # Volume rolling mean
    df_feat["vol_ma_3"] = df_feat["volume"].shift(1).rolling(3).mean()
    df_feat["vol_ma_7"] = df_feat["volume"].shift(1).rolling(7).mean()
    
    # Drop rows with NaN (first few rows due to lag/rolling)
    df_feat = df_feat.dropna().reset_index(drop=True)
    return df_feat

df_feat = create_time_series_features(df)

# Define features and target
target_col = "close"
feature_cols = [c for c in df_feat.columns if c not in ["date", target_col]]

X = df_feat[feature_cols]
y = df_feat[target_col]

# ==========================
# 3. Train/Test Split
# ==========================
split_idx = int(len(df_feat)*0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================
# 4. Train RandomForest
# ==========================
model = RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# Predict and evaluate
preds = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, preds)

# --- CORRECTED LINES FOR RMSE CALCULATION ---
# Calculate MSE first, then take the square root.
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse) 
# --- END CORRECTION ---

r2 = r2_score(y_test, preds)

print(f"Test MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

# ==========================
# 5. Plot Predictions vs Actual
# ==========================
plt.figure(figsize=(12,5))
plt.plot(df_feat["date"][split_idx:], y_test, label="Actual Close", color="blue")
plt.plot(df_feat["date"][split_idx:], preds, label="Predicted Close", color="red", alpha=0.7)
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("RandomForest Predictions with Time-Series Features")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# ==========================
# 6. Save Model & Artifacts
# ==========================
joblib.dump(model, "rf_close_model_ts.pkl")
joblib.dump(scaler, "scaler_ts.pkl")
artifacts = {
    "feature_columns": feature_cols,
    "full_raw_csv": "synthetic_market.csv"
}
joblib.dump(artifacts, "model_artifacts_ts.pkl")

print("✅ Model, scaler, and artifacts saved!")

# ==========================
# 1. Load Data
# ==========================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

df = pd.read_csv("synthetic_market.csv", parse_dates=["date"])


df = df.sort_values("date").reset_index(drop=True)

# Use close price (you can add volume later)
data = df[["close"]].values

# ==========================
# 2. Scale Data
# ==========================
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# ==========================
# 3. Create Sequences
# ==========================
def create_sequences(data, n_steps=10):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

n_steps = 10
X, y = create_sequences(data_scaled, n_steps)
X = X.reshape((X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)

# Train/test split
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# ==========================
# 4. Build LSTM Model
# ==========================
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
model.summary()

# ==========================
# 5. Train Model
# ==========================
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_data=(X_test, y_test), verbose=1)

# ==========================
# 6. Predictions & Evaluation
# ==========================
y_pred = model.predict(X_test)
# Inverse scale
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))
y_pred_inv = scaler.inverse_transform(y_pred)

# Plot
plt.figure(figsize=(12,5))
plt.plot(df["date"][split_idx+n_steps:], y_test_inv, label="Actual Close", color="blue")
plt.plot(df["date"][split_idx+n_steps:], y_pred_inv, label="Predicted Close", color="red", alpha=0.7)
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("LSTM Stock Price Prediction")
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Compute RMSE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae = mean_absolute_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)
print(f"LSTM Test MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

# ==========================
# 7. Save Model & Scaler
# ==========================
model.save("lstm_close_model.h5")
import joblib
joblib.dump(scaler, "lstm_scaler.pkl")
print("✅ LSTM model and scaler saved!")

# ==========================
# LSTM Forecast Next n Days
# ==========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

# --------------------------
# Load data, model, scaler
# --------------------------
df = pd.read_csv("synthetic_market.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

lstm_model = load_model("lstm_close_model.h5", compile=False)  # fix for H5 load issue
scaler = joblib.load("lstm_scaler.pkl")

# --------------------------
# Forecast function
# --------------------------
def forecast_next_days_lstm(df, n_days=5, n_steps=10):
    """
    Iteratively forecast next n_days of stock close prices using LSTM.
    
    df: DataFrame with 'close' and 'date' columns
    n_days: number of future days to forecast
    n_steps: number of past days LSTM uses for prediction
    """
    df_sorted = df.sort_values("date").reset_index(drop=True)
    close_prices = df_sorted["close"].values
    forecasted = []
    
    last_sequence = close_prices[-n_steps:]  # initial input for LSTM
    
    for day in range(1, n_days+1):
        # Scale last sequence
        seq_scaled = scaler.transform(last_sequence.reshape(-1,1)).reshape(1, n_steps, 1)
        
        # Predict next day
        pred_scaled = lstm_model.predict(seq_scaled, verbose=0)
        pred_close = scaler.inverse_transform(pred_scaled.reshape(-1,1))[0,0]
        
        # Append prediction
        last_date = df_sorted["date"].max()
        forecast_date = last_date + pd.Timedelta(days=day)
        forecasted.append({"date": forecast_date, "pred_close": pred_close})
        
        # Update sequence
        last_sequence = np.append(last_sequence[1:], pred_close)
    
    return pd.DataFrame(forecasted)

# --------------------------
# Run forecast & plot
# --------------------------
forecast_df = forecast_next_days_lstm(df, n_days=5, n_steps=10)
print(forecast_df)

plt.figure(figsize=(12,5))
plt.plot(df["date"], df["close"], label="Historical Close", color="blue")
plt.plot(forecast_df["date"], forecast_df["pred_close"], label="Forecasted Close", color="orange", marker='o')
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("LSTM Forecast for Next 5 Days")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
