from stockpredict.data_loader import fetch_stock_data, AVAILABLE_STOCKS
from stockpredict.features import create_features
from stockpredict.model import train_hybrid_model, evaluate_model, save_model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def run_cli():
    print("📈 StockPredict - Hybrid Regression Model")
    print("Available stocks:", list(AVAILABLE_STOCKS.keys()))
    symbol = input("Enter stock symbol from the list: ").upper()
    start = input("Enter start date (YYYY-MM-DD): ")
    end = input("Enter end date (YYYY-MM-DD): ")

    print(f"\nFetching data for {symbol}...")
    df = fetch_stock_data(symbol, start, end)

    df, feature_cols = create_features(df)
    X = df[feature_cols]
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("\nTraining hybrid model (Linear Regression + Random Forest)...")
    model = train_hybrid_model(X_train, np.ravel(y_train))

    rmse, preds = evaluate_model(model, X_test, y_test)
    print(f"✅ Model RMSE: {rmse:.2f}")

    save_model(model, f"{symbol}_model.pkl")
    print(f"Saved model to {symbol}_model.pkl")

    # show last 5 predictions
    actual_arr = np.asarray(y_test.values[-5:]).ravel()
    pred_arr = np.asarray(preds[-5:]).ravel()
    minlen = min(len(actual_arr), len(pred_arr))
    print("\nLast 5 Actual vs Predicted:")
    print(pd.DataFrame({"Actual": actual_arr[-minlen:], "Pred": pred_arr[-minlen:]}).to_string(index=False))

if __name__ == "__main__":
    run_cli()
# main.py (interactive CLI)
from stockpredict.data_loader import fetch_stock_data, AVAILABLE_STOCKS
from stockpredict.features import create_features
from stockpredict.model import train_hybrid_model, evaluate_model, save_model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def run_cli():
    print("📈 StockPredict - Hybrid Regression Model")
    print("Available stocks:", list(AVAILABLE_STOCKS.keys()))
    symbol = input("Enter stock symbol from the list: ").upper()
    start = input("Enter start date (YYYY-MM-DD): ")
    end = input("Enter end date (YYYY-MM-DD): ")

    print(f"\nFetching data for {symbol}...")
    df = fetch_stock_data(symbol, start, end)

    df, feature_cols = create_features(df)
    X = df[feature_cols]
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("\nTraining hybrid model (Linear Regression + Random Forest)...")
    model = train_hybrid_model(X_train, np.ravel(y_train))

    rmse, preds = evaluate_model(model, X_test, y_test)
    print(f"✅ Model RMSE: {rmse:.2f}")

    save_model(model, f"{symbol}_model.pkl")
    print(f"Saved model to {symbol}_model.pkl")

    # show last 5 predictions
    actual_arr = np.asarray(y_test.values[-5:]).ravel()
    pred_arr = np.asarray(preds[-5:]).ravel()
    minlen = min(len(actual_arr), len(pred_arr))
    print("\nLast 5 Actual vs Predicted:")
    print(pd.DataFrame({"Actual": actual_arr[-minlen:], "Pred": pred_arr[-minlen:]}).to_string(index=False))

if __name__ == "__main__":
    run_cli()
