from stockpredict.data_loader import fetch_stock_data, AVAILABLE_STOCKS
from stockpredict.features import create_features
from stockpredict.model import train_hybrid_model, evaluate_model, save_model
from sklearn.model_selection import train_test_split
import pandas as pd

def run_cli():
    print("Available stocks:", list(AVAILABLE_STOCKS.keys()))
    symbol = input("Enter stock symbol: ").upper()
    start = input("Enter start date (YYYY-MM-DD): ")
    end = input("Enter end date (YYYY-MM-DD): ")

    print(f"Fetching {symbol}...")
    df = fetch_stock_data(symbol, start, end)
    df, features = create_features(df)
    X = df[features]
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = train_hybrid_model(X_train, y_train)
    rmse, preds = evaluate_model(model, X_test, y_test)
    print(f"RMSE: {rmse:.2f}")

    save_model(model, f"{symbol}_model.pkl")
    print(f"Saved model to {symbol}_model.pkl")
    print(pd.DataFrame({"Actual": y_test.values[-5:], "Pred": preds[-5:]}))

if __name__ == "__main__":
    run_cli()
