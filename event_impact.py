import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# --- STEP 1: DEFINE EVENTS ---
fomc_dates = [
    "2023-02-01", 
    "2023-03-22", 
    "2023-05-03", 
    "2023-06-14", 
    "2023-07-26",
    "2023-09-20",
    "2023-11-01",
    "2023-12-13"
]

ticker = "GC=F" 
window = 5 

print(f"[INFO] Analyzing {ticker} reaction to {len(fomc_dates)} FOMC events...")

# --- STEP 2: GET MARKET DATA ---
df = yf.download(ticker, start="2023-01-01", end="2024-01-01", progress=False)

if df.empty:
    print("[FATAL] Download gagal. Cek internet.")
    exit()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

target_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
if target_col not in df.columns:
    print(f"[FATAL] Kolom {target_col} gak ketemu.")
    exit()

data = df[target_col]

# --- STEP 3: THE ENGINE (FIXED FOR PANDAS 2.0+) ---
event_responses = {}

for event_date in fomc_dates:
    try:
        target_date = pd.to_datetime(event_date)
        loc_idx = data.index.get_indexer([target_date], method='nearest')[0]
        if (loc_idx - window < 0) or (loc_idx + window + 1 > len(data)):
            print(f"[WARN] Event {event_date} terlalu mepet ujung data. Skip.")
            continue
        slice_data = data.iloc[loc_idx - window : loc_idx + window + 1]
        
        if len(slice_data) != (window * 2 + 1):
            print(f"[WARN] Data incomplete for {event_date}, skipping.")
            continue
            
        price_at_event = slice_data.iloc[window]
        normalized_return = (slice_data / price_at_event) - 1
        
        event_responses[event_date] = normalized_return.values

    except Exception as e:
        print(f"[ERROR] Processing {event_date}: {e}")

if not event_responses:
    print("[ERROR] Tidak ada event yang berhasil di-process.")
    exit()

plt.figure(figsize=(12, 7))
x_axis = range(-window, window + 1)

sum_returns = 0
count = 0

for date, returns in event_responses.items():
    plt.plot(x_axis, returns * 100, color='gray', alpha=0.3) 
    sum_returns += returns
    count += 1

avg_returns = (sum_returns / count) * 100
plt.plot(x_axis, avg_returns, color='red', linewidth=3, label='Average Reaction')

plt.axvline(x=0, color='black', linestyle='--', label='FOMC Decision Day')
plt.axhline(y=0, color='black', linewidth=0.5)
plt.title(f'Gold Price Reaction to FOMC Meetings (2023)\n(Window: +/- {window} Trading Days)')
plt.xlabel('Days Relative to Event')
plt.ylabel('Cumulative Return (%)')
plt.legend()
plt.grid(True, alpha=0.3)

print("[INFO] Event study complete. Plot generated.")
plt.show()