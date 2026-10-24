# EstateAI — Indian House Price Prediction

EstateAI uses **Multiple Linear Regression** trained on the uploaded Indian housing dataset (`dataset/train.csv`, 29,451 source records). The model predicts price in **lakh rupees** and the interface displays both rupees and lakhs.

## Run on Windows
1. Extract this ZIP to a normal folder.
2. Install Python 3.11 or 3.12 and enable **Add Python to PATH**.
3. Double-click `run_app.bat`.
4. The first run trains the EstateAI model from `dataset/train.csv`, then opens Streamlit (usually `http://localhost:8501`).

## Important: use the correct training script
Use `train_estateai_model.py` for the EstateAI prediction model. `train_model.py` is now a compatibility entry point that calls the same Indian-dataset training script. The old 545-row dataset has been removed.

## Manual run
```bat
python -m pip install -r requirements.txt
python train_estateai_model.py
python -m streamlit run app.py
```

## Model inputs
The prediction form matches the model's features: listing source, construction status, RERA, BHK/RK, area, ready-to-move status, resale status, and locality/city. Numeric imputation, scaling, categorical encoding, log-area feature engineering, and the inverse target transformation are applied consistently. The core estimator is still Multiple Linear Regression. The model fits `log(1 + price in lakhs)` and converts its output back to lakh rupees, which better handles the highly skewed house-price distribution.

Latest saved evaluation (`model/estateai_model_metrics.json`): held-out test R² **85.36%**, MAE **₹17.82 lakh**, RMSE **₹37.57 lakh**; five-fold training CV mean R² **76.81%** (standard deviation **5.16 percentage points**). These results use the fixed `random_state=123` split and the stated filters; they are dataset-specific and not a guarantee of real-world accuracy.

The old `Housing.csv` dataset (545 records) has been removed. Both prediction and analytics use the Indian housing research dataset in `dataset/train.csv`.

## Important
Predictions are estimates, not verified market valuations. Dataset location text and property records may be inconsistent, so results can be inaccurate for specific localities.
