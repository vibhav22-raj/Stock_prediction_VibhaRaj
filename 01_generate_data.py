# 01_generate_data.py
import os
import numpy as np
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ---- Code from Cell 1 (synthetic data generation) ----
# ✅ Paste your Cell 1 code here exactly as it is
# Cell 1: Generate synthetic share market dataset, preprocess, and split
import os
import numpy as np
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Config / seed
np.random.seed(42)
CSV_PATH = "synthetic_market.csv"
TRAIN_CSV = "market_train.csv"
TEST_CSV = "market_test.csv"

def generate_geometric_brownian(start_price=100.0, mu=0.0005, sigma=0.01, days=1500):
    """
    Generate a simple GBM price series (daily) with volume.
    mu: drift per day
    sigma: volatility per day
    """
    dt = 1.0
    prices = [start_price]
    for _ in range(days-1):
        prev = prices[-1]
        shock = np.random.normal(loc=(mu*dt), scale=(sigma*np.sqrt(dt)))
        new = prev * np.exp(shock)
        prices.append(new)
    return np.array(prices)

# Generate dates and series
days = 1500  # ~6 years of trading days (including weekends; synthetic)
start_date = pd.to_datetime("2019-01-01")
dates = pd.date_range(start_date, periods=days, freq="D")  # daily synthetic
close_prices = generate_geometric_brownian(start_price=100.0, mu=0.0006, sigma=0.02, days=days)

# Construct open/high/low from close with small random spreads, and volume
opens = close_prices * (1 + np.random.normal(0, 0.0015, size=days))
highs = np.maximum(opens, close_prices) * (1 + np.abs(np.random.normal(0, 0.005, size=days)))
lows = np.minimum(opens, close_prices) * (1 - np.abs(np.random.normal(0, 0.005, size=days)))
volumes = np.random.randint(100000, 2000000, size=days)

df = pd.DataFrame({
    "date": dates,
    "open": opens,
    "high": highs,
    "low": lows,
    "close": close_prices,
    "volume": volumes
})
df = df.sort_values("date").reset_index(drop=True)

# Save raw CSV
df.to_csv(CSV_PATH, index=False)
print(f"Saved raw synthetic series to {CSV_PATH}. Shape: {df.shape}")

# Preprocessing into supervised features
def make_features(df, lags=[1,2,3,5,7], rolling_windows=[3,7,14]):
    df_feat = df.copy()
    df_feat = df_feat.set_index("date")
    for lag in lags:
        df_feat[f"lag_{lag}"] = df_feat["close"].shift(lag)
    for w in rolling_windows:
        df_feat[f"roll_mean_{w}"] = df_feat["close"].rolling(window=w).mean().shift(1)
        df_feat[f"roll_std_{w}"] = df_feat["close"].rolling(window=w).std().shift(1)
    # Momentum / pct change features
    df_feat["pct_change_1"] = df_feat["close"].pct_change().shift(1)
    df_feat["vol_ma_7"] = df_feat["volume"].rolling(window=7).mean().shift(1)
    # target: next-day close
    df_feat["target_close"] = df_feat["close"].shift(-1)
    df_feat = df_feat.dropna().reset_index()
    return df_feat

df_feat = make_features(df)
print("Feature dataframe shape:", df_feat.shape)
df_feat.head()
# Save processed dataset (optional)
df_feat.to_csv("synthetic_market_features.csv", index=False)

# Time-based train/test split (last 20% as test)
split_index = int(len(df_feat) * 0.8)
train_df = df_feat.iloc[:split_index].reset_index(drop=True)
test_df = df_feat.iloc[split_index:].reset_index(drop=True)

train_df.to_csv(TRAIN_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)
print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

# Quick plot of the close prices
plt.figure(figsize=(10,4))
plt.plot(df['date'], df['close'], label="Close")
plt.title("Synthetic Close Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.tight_layout()
plt.show()
