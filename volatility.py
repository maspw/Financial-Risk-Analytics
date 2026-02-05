import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta # Kita butuh ini buat ngitung tanggal hari ini

# --- STEP 1: DATA INGESTION (FULL INTERACTIVE) ---
print("\n=== VOLATILITY ANALYZER MACHINE ===")
print("Masukan Ticker Symbol (Contoh: BBCA.JK, NVDA, BTC-USD)")
ticker = input("Ticker >> ").upper()

# Default: End Date = Hari Ini, Start Date = 1 Tahun Lalu
today = datetime.today().strftime('%Y-%m-%d')
last_year = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

print(f"\nMasukan Tanggal (Format: YYYY-MM-DD). Tekan ENTER untuk Default.")
start_input = input(f"Start Date [Default: {last_year}] >> ")
end_input = input(f"End Date   [Default: {today}] >> ")

# Logic: Kalau user tekan Enter (kosong), pake default
start_date = start_input if start_input else last_year
end_date = end_input if end_input else today

print(f"\n[INFO] Fetching data for {ticker} ({start_date} to {end_date})...")
df = yf.download(ticker, start=start_date, end=end_date, progress=False)
# --- STEP 2: CALCULATE RETURNS ---
try:
    target_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    df['Log_Ret'] = np.log(df[target_col] / df[target_col].shift(1))
except KeyError:
    print(f"[FATAL] Gak nemu kolom harga. Struktur data: {df.columns}")
    exit()

df.dropna(inplace=True) 

# --- STEP 3: VOLATILITY MODELS ---
ann_factor = np.sqrt(252)

# Model A: Rolling 20
df['Vol_Roll_20'] = df['Log_Ret'].rolling(window=20).std() * ann_factor

# Model B: EWMA 20
df['Vol_EWMA_20'] = df['Log_Ret'].ewm(span=20).std() * ann_factor

# --- STEP 4: VISUALIZATION ---
print(f"[INFO] Generating plot for {ticker}...")
plt.figure(figsize=(14, 7))

# Plot Harga
ax1 = plt.subplot(2, 1, 1)
ax1.plot(df.index, df[target_col], color='black', alpha=0.6)
ax1.set_title(f'Price Action: {ticker}')
ax1.grid(True, alpha=0.3)

# Plot Volatilitas
ax2 = plt.subplot(2, 1, 2, sharex=ax1) 
ax2.plot(df.index, df['Vol_Roll_20'], label='Rolling 20D (Lagging)', linestyle='--', color='blue')
ax2.plot(df.index, df['Vol_EWMA_20'], label='EWMA 20D (Reactive)', color='red', linewidth=2)

ax2.set_title(f'Volatility Regimes: {ticker}')
ax2.set_ylabel('Annualized Volatility')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
print("[INFO] Plot displayed.")
plt.show()