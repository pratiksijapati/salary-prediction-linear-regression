# Salary Prediction with Linear Regression

Predicts salary from years of experience using linear regression, with a small Streamlit app for interactive predictions.

> **Status:** small machine-learning exercise.

## What it does

- `train.py` loads `data/salary_data.csv` (downloading a public dataset if missing), splits train/test, scales the feature, fits `LinearRegression` and saves the model plus R², MSE and MAE to `metrics.json`.
- `streamlit_app.py` shows the metrics and predicts a salary for the years of experience you enter.

## Tech stack

Python · scikit-learn · pandas · Streamlit · joblib

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run streamlit_app.py
```
