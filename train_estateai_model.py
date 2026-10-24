"""Train EstateAI's Multiple Linear Regression model on the Indian housing dataset."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset" / "train.csv"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "estateai_indian_house_model.pkl"
METRICS_PATH = MODEL_DIR / "estateai_model_metrics.json"
NUMERIC = ["UNDER_CONSTRUCTION", "RERA", "BHK_NO.", "SQUARE_FT",
           "READY_TO_MOVE", "RESALE", "SQFT_PER_BHK", "LOG_SQUARE_FT"]
CATEGORICAL = ["POSTED_BY", "BHK_OR_RK", "LOCALITY", "LOCATION_FIELD"]

def prepare(df):
    data = df.copy()
    data["LOCALITY"] = data["ADDRESS"].fillna("").astype(str).str.split(",").str[0].str.strip()
    data["LOCATION_FIELD"] = data["ADDRESS"].fillna("").astype(str).str.split(",").str[-1].str.strip()
    data["SQFT_PER_BHK"] = data["SQUARE_FT"] / data["BHK_NO."].replace(0, np.nan)
    data["SQFT_PER_BHK"] = data["SQFT_PER_BHK"].replace([np.inf, -np.inf], np.nan)
    data["LOG_SQUARE_FT"] = np.log1p(data["SQUARE_FT"].clip(lower=0))
    return data[NUMERIC + CATEGORICAL]

def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Missing dataset: {DATASET}")
    df = pd.read_csv(DATASET)
    target = "TARGET(PRICE_IN_LACS)"
    df = df[df["SQUARE_FT"].between(100, 10000) & df[target].between(0.01, 1000)].copy()
    X = prepare(df)
    y = df[target].astype(float)
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), NUMERIC),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL)
    ])
    base_model = Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])
    # House prices are strongly right-skewed. Fit MLR to log(1 + price), then
    # convert predictions back to lakh rupees. The underlying algorithm remains MLR.
    model = TransformedTargetRegressor(regressor=base_model, func=np.log1p, inverse_func=np.expm1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    # Cross-validation is performed only on the training split; the held-out test set
    # remains separate for the final reported metrics.
    cv = KFold(n_splits=5, shuffle=True, random_state=123)
    from sklearn.base import clone
    cv_scores = []
    for train_idx, valid_idx in cv.split(X_train):
        fold_model = clone(model)
        fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        cv_scores.append(r2_score(y_train.iloc[valid_idx], fold_model.predict(X_train.iloc[valid_idx])))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    metrics = {
        "algorithm": "Multiple Linear Regression with log-transformed target",
        "dataset_rows_after_filtering": int(len(df)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "cv_r2_mean": float(np.mean(cv_scores)),
        "cv_r2_std": float(np.std(cv_scores, ddof=1)),
        "test_r2": float(r2_score(y_test, pred)),
        "test_mae_lakh": float(mean_absolute_error(y_test, pred)),
        "test_rmse_lakh": float(np.sqrt(mean_squared_error(y_test, pred))),
        "target_transform": "log1p; inverse transform expm1"
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Training rows: {len(X_train)} | test rows: {len(X_test)}")
    print(f"5-fold training CV R2: {metrics['cv_r2_mean']:.4f} +/- {metrics['cv_r2_std']:.4f}")
    print(f"MAE: {mean_absolute_error(y_test, pred):.2f} lakh")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.2f} lakh")
    print(f"R2: {r2_score(y_test, pred):.4f}")
    print(f"Saved model: {MODEL_PATH}")

if __name__ == "__main__":
    main()
