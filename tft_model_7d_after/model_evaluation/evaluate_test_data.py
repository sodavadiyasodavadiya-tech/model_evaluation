import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import TFTModel
import json
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Ensure plots are saved without needing a display
import matplotlib
matplotlib.use('Agg')

def evaluate_test_dataset():
    # --- 1. CONFIGURATION ---
    WEB_ID = 'qb1146021'
    TRAIN_DATA_PATH = 'prepared_data_with_target.csv'
    TEST_DATA_PATH = 'test_data.csv'
    CONFIG_PATH = 'tft_model_after_7d/training_config.json'
    MODEL_DIR = 'tft_model_after_7d'
    CHECKPOINT_FILE = 'best-epoch=8-val_loss=1.04.ckpt' # Based on list_dir
    WORK_DIR = '.'

    print(f"Loading configuration from {CONFIG_PATH}...")
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    # Extract parameters from config
    target_col = config['target_column']
    past_covs_cols = config.get('past_covariates', [])
    fut_covs_cols = config.get('future_covariates', [])
    stat_covs_cols = config.get('static_covariates', [])
    icl = config['model_parameters']['input_chunk_length']
    ocl = config['model_parameters']['output_chunk_length']

    # --- 2. LOAD MODEL ---
    print(f"Loading TFT model from {MODEL_DIR}...")
    # We use the checkpoint identified in the directory
    # Darts automatically looks into the 'checkpoints' subfolder
    model = TFTModel.load_from_checkpoint(
        model_name=MODEL_DIR, 
        work_dir=WORK_DIR, 
        file_name=CHECKPOINT_FILE
    )
    model.model.eval() # Set to evaluation mode

    # --- 3. LOAD DATASETS ---
    print(f"Loading training dataset to extract web_id: {WEB_ID}...")
    df_train_full = pd.read_csv(TRAIN_DATA_PATH)
    df_train_full['date'] = pd.to_datetime(df_train_full['date'])
    df_train = df_train_full[df_train_full['web_id'] == WEB_ID].copy().sort_values('date')

    print(f"Loading test dataset from {TEST_DATA_PATH}...")
    df_test_full = pd.read_csv(TEST_DATA_PATH)
    df_test_full['date'] = pd.to_datetime(df_test_full['date'])
    
    # Filter test data for the same web_id
    df_test = df_test_full[df_test_full['web_id'] == WEB_ID].copy().sort_values('date')

    if df_test.empty:
        print(f"Error: No data found for web_id {WEB_ID} in test dataset.")
        return

    # Handle numeric conversion and missing values
    all_cols = [target_col] + past_covs_cols + fut_covs_cols + stat_covs_cols
    for col in all_cols:
        if col in df_test.columns:
            df_test[col] = pd.to_numeric(df_test[col], errors='coerce').fillna(0)
        if col in df_train.columns:
            df_train[col] = pd.to_numeric(df_train[col], errors='coerce').fillna(0)

    # --- 4. PREPARE TIMESERIES OBJECTS ---
    # To use historical_forecasts on the test set, we often need a bit of history (icl) 
    # to make the first prediction. We'll join a small tail of the training data 
    # to the test data to ensure the model has enough context if the test data 
    # doesn't already provide it.
    
    # Check if test data has enough history by itself
    # But according to instructions, we should just prepare the test series.
    # If the gap between train and test is large, we might only be able to 
    # start forecasting after 'icl' steps into the test data.
    
    print("Preparing TimeSeries objects...")
    
    # It's safer to combine them if they are continuous, but here there is a gap.
    # Darts historical_forecasts handles sequences.
    
    # Create test TimeSeries
    # Filter only available columns to avoid KeyError, but the model expects the exact set it was trained with.
    available_past = [c for c in past_covs_cols if c in df_test.columns]
    available_fut = [c for c in fut_covs_cols if c in df_test.columns]
    
    print(f"Past covariates: {len(available_past)} expected: {len(past_covs_cols)}")
    print(f"Future covariates: {len(available_fut)} expected: {len(fut_covs_cols)}")
    
    if len(available_past) < len(past_covs_cols):
        missing = set(past_covs_cols) - set(available_past)
        print(f"CRITICAL: Missing past covariates from test_data.csv: {missing}")
        # Filling missing with zeros to keep the model happy
        for c in missing:
            df_test[c] = 0
            available_past.append(c)
            
    if len(available_fut) < len(fut_covs_cols):
        missing = set(fut_covs_cols) - set(available_fut)
        print(f"CRITICAL: Missing future covariates from test_data.csv: {missing}")
        for c in missing:
            df_test[c] = 0
            available_fut.append(c)

    target_ts = TimeSeries.from_dataframe(df_test, time_col='date', value_cols=target_col, static_covariates=df_test[stat_covs_cols].iloc[[0]])
    past_covs_ts = TimeSeries.from_dataframe(df_test, time_col='date', value_cols=past_covs_cols) if past_covs_cols else None
    fut_covs_ts = TimeSeries.from_dataframe(df_test, time_col='date', value_cols=fut_covs_cols) if fut_covs_cols else None

    # --- 5. GENERATE FORECASTS ---
    print(f"Generating rolling forecasts for the test dataset (horizon={ocl})...")
    
    # historical_forecasts will start at the earliest possible point (after icl steps)
    forecasts = model.historical_forecasts(
        series=target_ts,
        past_covariates=past_covs_ts,
        future_covariates=fut_covs_ts,
        start=None, # Start at the first possible point
        forecast_horizon=ocl,
        retrain=False,
        last_points_only=True,
        verbose=False
    )

    # --- 6. COMPUTE EVALUATION METRICS ---
    print("Computing evaluation metrics...")
    
    # Match indices for comparison
    common_dates = target_ts.time_index.intersection(forecasts.time_index)
    if len(common_dates) == 0:
        print("Error: No overlapping dates for evaluation. Check if test data length > input_chunk_length.")
        return

    y_true = target_ts.slice(common_dates[0], common_dates[-1]).values().flatten()
    y_pred = forecasts.slice(common_dates[0], common_dates[-1]).values().flatten()

    print(f"y_true (first 10 values): {y_true[:10]}")
    print(f"y_pred (first 10 values): {y_pred[:10]}")

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    sum_abs_true = np.sum(np.abs(y_true))
    wape = np.sum(np.abs(y_true - y_pred)) / sum_abs_true if sum_abs_true != 0 else np.nan

    print("\n" + "="*40)
    print(f"Evaluation Results for Test Dataset (web_id: {WEB_ID}):")
    print("="*40)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"WAPE: {wape:.4f}")
    print("="*40 + "\n")

    # --- 7. VISUALIZE RESULTS ---
    print("Generating performance plot...")
    plt.figure(figsize=(14, 7))
    plt.plot(common_dates, y_true, label='Actual Sales', color='blue', marker='o', alpha=0.7)
    plt.plot(common_dates, y_pred, label='Predicted Sales', color='red', linestyle='--', marker='x', alpha=0.9)
    plt.title(f'TFT Model Performance on Test Data (web_id: {WEB_ID})')
    plt.xlabel('Date')
    plt.ylabel('next_7d_sales')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    output_plot = 'evaluation_performance_test.png'
    plt.savefig(output_plot)
    plt.close()
    
    print(f"Evaluation completed. Plot saved as {output_plot}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    evaluate_test_dataset()
