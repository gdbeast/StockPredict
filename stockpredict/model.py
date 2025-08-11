from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

def train_hybrid_model(X_train, y_train):
    if hasattr(y_train, "ravel"):
        y_train = np.ravel(y_train)

    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    stack = StackingRegressor(
        estimators=[('lr', lr), ('rf', rf)],
        final_estimator=LinearRegression()
    )
    stack.fit(X_train, y_train)
    return stack

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    preds = np.asarray(preds).ravel()   
    mse = mean_squared_error(np.asarray(y_test).ravel(), preds)
    rmse = np.sqrt(mse)
    return rmse, preds

def save_model(model, path="model.pkl"):
    joblib.dump(model, path)

def load_model(path="model.pkl"):
    return joblib.load(path)
