Financial Risk Analytics Suite 📊

A quantitative toolkit built with Python to analyze asset volatility, portfolio correlations, and market anomalies surrounding macroeconomic events (e.g., FOMC meetings).

🚀 Key Modules

 1. Volatility Estimator (`machine.py`)

Calculates and compares risk metrics:

-  Rolling Volatility:   Traditional historical standard deviation.
-   EWMA (Exponentially Weighted Moving Average):   Reactive volatility model that assigns higher weight to recent price shocks.

    2. Portfolio Correlation (`heatmap.py`)

Dynamic analysis of asset relationships:

- Generates correlation matrices for multi-asset portfolios (Equities, Forex, Crypto, Commodities).
- Visualizes diversification breakdowns using Seaborn heatmaps.

    3. Event Impact Analyzer (`event_impact.py`)

Event-study framework to quantify market behavior:

- Analyzes price drift (Pre-Event) and reaction (Post-Event) around scheduled economic releases.
- Normalized returns to visualize aggregate market sentiment.

   🛠 Tech Stack

-   Core:   Python 3.10+
-   Data:   `yfinance` (Yahoo Finance API)
-   Computation:   `pandas`, `numpy`
-   Visualization:   `matplotlib`, `seaborn`

   📦 How to Run

1. Clone the repo:
   ```bash
   git clone [https://github.com/username/Financial-Risk-Analytics.git](https://github.com/maspw/Financial-Risk-Analytics.git)
   ```
