# run_colab.py - Colab-friendly script (no display() usage, defensive shapes)
SYMBOL = "RELIANCE"
START = "2023-01-01"
END = "2024-12-31"

from stockpredict.data_loader import fetch_stock_data, AVAILABLE_STOCKS
from stockpredict.features import create_features
from stockpredict.model import train_hybrid_model, evaluate_model, save_model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def main():
    print("Available:", list(AVAILABLE_STOCKS.keys()))
    print("Running for", SYMBOL, START, END)

    df = fetch_stock_data(SYMBOL, START, END)
    df, features = create_features(df)
    X = df[features]
    y = df['Close']

    # time-based split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # ensure y_train is 1-D array before training to avoid sklearn warning
    y_train = np.ravel(y_train)

    model = train_hybrid_model(X_train, y_train)
    rmse, preds = evaluate_model(model, X_test, y_test)
    print(f"RMSE: {rmse:.2f}")

    save_model(model, f"{SYMBOL}_model.pkl")
    print("Saved model:", f"{SYMBOL}_model.pkl")

    # Defensive preparation for printing (ensure 1-D and same length)
    actual_arr = np.asarray(y_test.values[-10:]).ravel()
    pred_arr   = np.asarray(preds[-10:]).ravel()

    # Align sizes
    minlen = min(len(actual_arr), len(pred_arr))
    actual_arr = actual_arr[-minlen:]
    pred_arr   = pred_arr[-minlen:]

    comp = pd.DataFrame({
        "Actual": actual_arr,
        "Pred":   pred_arr
    })
    print("\nLast predictions vs actuals:")
    print(comp.to_string(index=False))

if __name__ == "__main__":
    main()
