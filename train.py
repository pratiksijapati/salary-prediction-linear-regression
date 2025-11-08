"""
Train a simple Linear Regression on Salary vs YearsExperience.
Artifacts: supervised/model.joblib, supervised/scaler.joblib, supervised/metrics.json
"""
import os, json, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "data", "salary_data.csv")
MODEL_PATH = os.path.join(HERE, "model.joblib")
SCALER_PATH = os.path.join(HERE, "scaler.joblib")
METRICS_PATH = os.path.join(HERE, "metrics.json")

def ensure_data():
    if os.path.exists(DATA_PATH):
        return
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    import urllib.request
    url = ("https://raw.githubusercontent.com/sudarshan-koirala/"
           "Salary-Prediciton-based-on-Years-of-Experience/refs/heads/master/Salary_Data.csv")
    urllib.request.urlretrieve(url, DATA_PATH)
    print(f"Downloaded dataset to {DATA_PATH}")

def main():
    ensure_data()
    df = pd.read_csv(DATA_PATH)
    X = df[["YearsExperience"]].values
    y = df["Salary"].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    r2 = r2_score(y_te, y_pred)
    mse = mean_squared_error(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({"r2": r2, "mse": mse, "mae": mae}, f, indent=2)

    print(f"Saved model → {MODEL_PATH}")
    print(f"R2: {r2:.3f}  MSE: {mse:.1f}  MAE: {mae:.1f}")

if __name__ == "__main__":
    main()
