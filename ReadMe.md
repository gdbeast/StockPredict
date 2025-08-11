StockPredict — Hybrid Regression for NSE Stocks
StockPredict is a lightweight, GitHub-ready Python project that demonstrates a hybrid regression approach (Linear Regression + Random Forest) to predict closing prices for NSE-listed tickers.
This repo is designed to be easy to run, easy to extend, and interview-friendly — it implements the features shown on the resume: hybrid regression + feature-driven analysis (trend, regularity, seasonality, volatility).

Features
Hybrid model: Linear Regression (baseline) + Random Forest (captures non-linear patterns) combined using stacking.

Feature-driven pipeline:

Trend: previous close (lag1)

Regularity: moving average (ma5)

Volatility: rolling std (vol5)

Seasonality: day-of-week encoded as sin/cos

Works for multiple NSE stocks via Yahoo Finance .NS tickers (menu-driven).

CLI (main.py) for local use; run_colab.py for non-interactive Colab runs.

Saves trained model as SYMBOL_model.pkl (using joblib).

Repo structure
bash
Copy
Edit
StockPredict/
├── main.py                # CLI entrypoint (interactive)
├── run_colab.py           # Non-interactive script for Colab
├── requirements.txt
├── .gitignore
├── README.md
└── stockpredict/
    ├── __init__.py
    ├── data_loader.py    # download stock data (yfinance)
    ├── features.py       # feature engineering (lag, ma, vol, dow sin/cos)
    └── model.py          # train/eval/save/load hybrid model
Requirements
Tested with Python 3.8+. Install dependencies:

bash
Copy
Edit
python -m venv venv
# mac / linux
source venv/bin/activate
# windows
# .\venv\Scripts\activate

pip install -r requirements.txt
requirements.txt should include:

nginx
Copy
Edit
pandas
numpy
yfinance
scikit-learn
joblib
Quick start — run locally (interactive)
Launch the CLI:

bash
Copy
Edit
python main.py
Follow prompts:

Enter a stock symbol from the printed list (e.g. RELIANCE)

Enter start date and end date in YYYY-MM-DD format

The script:

fetches historical data,

builds features,

splits train/test (time-ordered, no shuffle),

trains hybrid model (stacked LR+RF),

prints RMSE and last few Predicted vs Actual values,

saves a model file: SYMBOL_model.pkl.

Run in Google Colab (non-interactive)
Open a new Colab notebook.


python
Copy
Edit
!git clone https://github.com/gdbeast/StockPredict.git
%cd StockPredict
!ls -la
Install dependencies:

python
Copy
Edit
!pip install -r requirements.txt
Run the Colab script (edit run_colab.py top variables if needed):

python
Copy
Edit
!python run_colab.py
run_colab.py includes top-level variables:

python
Copy
Edit
SYMBOL = "RELIANCE"
START = "2023-01-01"
END   = "2024-12-31"
Change them in the file or open the file in Colab and edit before running.

Example output
After running, expect output like:

sql
Copy
Edit
Available: ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'SBIN']
Running for RELIANCE 2023-01-01 2024-12-31
RMSE: 18.72
Saved model: RELIANCE_model.pkl
   Actual     Pred
... (last rows) ...
RMSE will vary by ticker & date range.

Add more stocks
Open stockpredict/data_loader.py and update AVAILABLE_STOCKS:

python
Copy
Edit
AVAILABLE_STOCKS = {
  "RELIANCE": "RELIANCE.NS",
  "TCS": "TCS.NS",
  "INFY": "INFY.NS",
  "HDFCBANK": "HDFCBANK.NS",
  "SBIN": "SBIN.NS",
  "NEW": "NEW.NS"   # add new ticker here
}
Use Yahoo Finance to find the correct ticker (append .NS).

Notes, caveats & tips
Data source: yfinance is used for convenience (Yahoo Finance .NS tickers). For production or higher-fidelity data use paid/official market data APIs.

Prediction limits: This prototype is educational — not financial advice.

Model improvements: You can extend features (RSI, MACD), try different stacking strategies, or use time-series specific models (ARIMA, Prophet, LSTM).

Model serving: To create a /predict endpoint later, add a FastAPI app that loads SYMBOL_model.pkl and serves JSON predictions.

Versioning: Save model metadata (training dates, performance metrics) alongside the model for reproducibility.

Suggested next steps (good for interviews)
Add time-series cross-validation (walk-forward validation).

Save model and feature pipeline together (use sklearn.pipeline or joblib).

Add hyperparameter tuning (GridSearchCV or RandomizedSearchCV).

Build a minimal FastAPI /predict endpoint and dockerize.

Add a Streamlit dashboard to visualize predictions vs actuals.

License
This project is provided as a learning sample. Add a license file if you want to publish (e.g., MIT).

Contact / Author
If you want, I can:

create a ready-to-run Colab notebook (.ipynb) from run_colab.py, or

add a FastAPI predict endpoint and Dockerfile, or

add unit tests and a simple CI using GitHub Actions.

Pick one and I’ll provide the full code for it.