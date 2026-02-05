import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# --- STEP 1: DYNAMIC INPUT ---
print("\n=== PORTFOLIO CORRELATION MATRIX ===")
print("Masukan Daftar Ticker (Pisahkan dengan KOMA).")
print("Contoh: BBCA.JK, BBRI.JK, TLKM.JK, GC=F, ^GSPC")

# 1. Input Ticker String
raw_input = input("Tickers >> ").upper()

# 2. Parsingn
tickers = [t.strip() for t in raw_input.split(',')]

# Validasi minimal 2 aset
if len(tickers) < 2:
    print("[ERROR] Butuh minimal 2 ticker buat hitung korelasi bro.")
    exit()

# 3. Input Tanggal (Sama kayak machine.py)
today = datetime.today().strftime('%Y-%m-%d')
last_year = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

print(f"\nMasukan Tanggal (Format: YYYY-MM-DD). Enter untuk Default.")
start_input = input(f"Start Date [Default: {last_year}] >> ")
end_input = input(f"End Date   [Default: {today}]     >> ")

start_date = start_input if start_input else last_year
end_date = end_input if end_input else today

print(f"\n[INFO] Fetching data for: {tickers}...")
raw_df = yf.download(tickers, start=start_date, end=end_date, progress=False)

if raw_df.empty:
    print("[FATAL] Download gagal total. Cek ticker atau internet.")
    exit()

# --- STEP 2: DATA CLEANING ---
# Target Price: Adj Close > Close
target_col = 'Adj Close'

try:
    # Handle MultiIndex Columns (Ribetnya yfinance kalau download banyak)
    if isinstance(raw_df.columns, pd.MultiIndex):
        # Cek level 0 ada 'Adj Close' gak
        if target_col in raw_df.columns.get_level_values(0):
            data = raw_df[target_col]
        elif 'Close' in raw_df.columns.get_level_values(0):
            print(f"[WARN] '{target_col}' gak ada, pake 'Close'.")
            data = raw_df['Close']
        else:
            print("[FATAL] Struktur data aneh, gak nemu harga.")
            exit()
    else:
        # Kalau cuma download 1 ticker (kasus user iseng input 1 doang tapi lolos validasi)
        data = raw_df[target_col] if target_col in raw_df.columns else raw_df['Close']

except Exception as e:
    print(f"[ERROR] Data processing failed: {e}")
    exit()

# Bersihin NaN (Wajib buat korelasi)
data = data.dropna()
if data.empty:
    print("[ERROR] Setelah dibersihin datanya kosong (mungkin beda hari libur bursa).")
    exit()

# --- STEP 3: CORRELATION MATH ---
print(f"[INFO] Calculating correlation on {len(data)} common trading days...")
returns = data.pct_change().dropna()
corr_matrix = returns.corr()

# Tunjukin Top Correlation di Terminal (Quick View)
print("\n--- TOP CORRELATIONS (Strongest to Weakest) ---")
try:
    sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                      .stack()
                      .sort_values(ascending=False))
    print(sol.head(5))
except:
    print("Not enough data to rank.")

# --- STEP 4: VISUALIZATION ---
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap='coolwarm', 
            vmin=-1, vmax=1, 
            square=True,
            linewidths=.5)

plt.title(f'Correlation Matrix\n({start_date} to {end_date})')
plt.tight_layout()
print("[INFO] Heatmap deployed.")
plt.show()