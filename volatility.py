import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def main():
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

    # Validasi input tanggal
    try:
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        print("[ERROR] Format tanggal salah. Harus YYYY-MM-DD.")
        return

    print(f"\n[INFO] Fetching data for {ticker} ({start_date} to {end_date})...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if df.empty:
        print(f"[ERROR] Data tidak ditemukan untuk {ticker}. Periksa ticker atau tanggal.")
        return

    # --- STEP 2: CALCULATE RETURNS ---
    try:
        target_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        df['Log_Ret'] = np.log(df[target_col] / df[target_col].shift(1))
    except KeyError:
        print(f"[FATAL] Gak nemu kolom harga. Struktur data: {df.columns}")
        return

    df.dropna(inplace=True)

    # --- STEP 3: ADDITIONAL INDICATORS (RSI & SMA) ---
    # SMA 50
    df['SMA_50'] = df[target_col].rolling(window=50).mean()

    # RSI 14 (Wilder's Smoothing)
    delta = df[target_col].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()

    # Calculate Wilder's smoothed moving average
    for i in range(14, len(df)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * 13 + gain.iloc[i]) / 14
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * 13 + loss.iloc[i]) / 14

    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # --- STEP 4: VOLATILITY MODELS ---
    ann_factor = np.sqrt(252)

    # Model A: Rolling 20
    df['Vol_Roll_20'] = df['Log_Ret'].rolling(window=20).std() * ann_factor

    # Model B: EWMA 20
    df['Vol_EWMA_20'] = df['Log_Ret'].ewm(span=20).std() * ann_factor

    # --- STEP 5: DECISION SUPPORT SYSTEM (DSS) LOGIC ---
    if len(df) < 50:
        recommendation = "N/A - DATA INSUFFICIENT"
        rec_color = 'gray'
        terminal_color = '\033[90m' # Dark Gray
    else:
        last_price = df[target_col].iloc[-1]
        last_sma50 = df['SMA_50'].iloc[-1]
        last_rsi14 = df['RSI_14'].iloc[-1]
        last_vol_ewma20 = df['Vol_EWMA_20'].iloc[-1]

        if last_vol_ewma20 > 0.50:
            recommendation = "AVOID"
            rec_color = 'red'
            terminal_color = '\033[91m' # Red
        elif last_price > last_sma50 and last_rsi14 < 60:
            recommendation = "BUY"
            rec_color = 'green'
            terminal_color = '\033[92m' # Green
        elif last_rsi14 > 75 or last_price < last_sma50:
            recommendation = "SELL"
            rec_color = 'red'
            terminal_color = '\033[91m' # Red
        else:
            recommendation = "HOLD"
            rec_color = 'gold' # Yellow equivalent for matplotlib that's easier to read
            terminal_color = '\033[93m' # Yellow

    reset_color = '\033[0m'

    # --- STEP 6: TERMINAL OUTPUT ---
    box_border = "=" * 40
    print("\n" + box_border)
    if recommendation == "N/A - DATA INSUFFICIENT":
        print(f"[WARNING] Data kurang dari 50 hari. Tidak bisa menganalisis.")
    print(f"[DECISION] Rekomendasi untuk {ticker}:")
    print(f"{terminal_color}>> {recommendation} <<{reset_color}")
    print(box_border + "\n")

    # --- STEP 7: VISUALIZATION ---
    print(f"[INFO] Generating plot for {ticker}...")
    plt.figure(figsize=(14, 7))

    # Plot Harga
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df.index, df[target_col], color='black', alpha=0.6)

    # Add SMA 50 line
    ax1.plot(df.index, df['SMA_50'], color='orange', label='SMA 50', alpha=0.8)

    ax1.set_title(f'Price Action: {ticker}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Teks Rekomendasi di Pojok Kanan Atas
    ax1.text(0.98, 0.95, f'DECISION: {recommendation}',
             transform=ax1.transAxes,
             fontsize=14, fontweight='bold',
             color=rec_color,
             horizontalalignment='right',
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor=rec_color, boxstyle='round,pad=0.5'))

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

if __name__ == '__main__':
    main()
