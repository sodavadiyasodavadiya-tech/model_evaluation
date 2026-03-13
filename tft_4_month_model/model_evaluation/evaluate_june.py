import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import TFTModel
import json
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_june_data():
    WEB_ID = 'qb1146021'
    DATA_PATH = 'prepared_data_with_target.csv'
    CONFIG_PATH = 'tft_model_4_month/training_config.json'
    MODEL_NAME = 'tft_model_4_month'
    WORK_DIR = '.'

    print(f"Loading configuration from {CONFIG_PATH}...")
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    # Use the first selected product from config if available
    if 'selected_products' in config and len(config['selected_products']) > 0:
        WEB_ID = config['selected_products'][0]
    
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])

    print(f"Filtering data for web_id: {WEB_ID}...")
    df_web = df[df['web_id'] == WEB_ID].copy()
    df_web = df_web.sort_values('date')

    # Define Evaluation period for June
    # Data is split chronologically:
    # Training Data used to be Feb-May
    # Evaluation Data is June
    eval_start = pd.Timestamp('2025-06-01')
    eval_end = pd.Timestamp('2025-06-30')

    # We keep data up to eval_end so that the model can use pre-June data for input_chunk sequences
    df_web = df_web[df_web['date'] <= eval_end]
    
    print(f"Total rows for {WEB_ID} up to {eval_end.date()}: {len(df_web)}")

    # Extract feature lists
    past_covariates_cols = config.get('past_covariates', [])
    future_covariates_cols = config.get('future_covariates', [])
    static_covariates_cols = config.get('static_covariates', [])
    target_col = config['target_column']

    # Ensure numeric types
    for col in past_covariates_cols + future_covariates_cols + static_covariates_cols + [target_col]:
        if col in df_web.columns:
            df_web[col] = pd.to_numeric(df_web[col], errors='coerce').fillna(0)

    # Prepare static covariates DataFrame (extract for the web_id)
    static_covs_df = None
    if static_covariates_cols:
        static_covs_df = df_web[static_covariates_cols].iloc[[0]]

    # Create TimeSeries objects
    print("Converting to Darts TimeSeries objects...")
    target_series = TimeSeries.from_dataframe(
        df_web, 
        time_col='date', 
        value_cols=target_col,
        static_covariates=static_covs_df
    )
    
    past_cov_series = None
    if past_covariates_cols:
        past_cov_series = TimeSeries.from_dataframe(df_web, time_col='date', value_cols=past_covariates_cols)
        
    future_cov_series = None
    if future_covariates_cols:
        future_cov_series = TimeSeries.from_dataframe(df_web, time_col='date', value_cols=future_covariates_cols)

    # Load previously trained model architecture and weights
    print(f"Loading TFT model '{MODEL_NAME}' from checkpoint...")
    try:
        model = TFTModel.load_from_checkpoint(
            model_name=MODEL_NAME, 
            work_dir=WORK_DIR, 
            file_name="best-epoch=7-val_loss=1.43.ckpt"
        )
    except Exception as e:
        print(f"Failed to load using file_name, falling back to best=True. Reason: {e}")
        model = TFTModel.load_from_checkpoint(
            model_name=MODEL_NAME, 
            work_dir=WORK_DIR, 
            best=True
        )

    forecast_horizon = 7
    print(f"Running rolling evaluation for June (start={eval_start.date()}, forecast_horizon={forecast_horizon})...")
    
    # Generate rolling predictions
    forecasts = model.historical_forecasts(
        series=target_series,
        past_covariates=past_cov_series,
        future_covariates=future_cov_series,
        start=eval_start,
        forecast_horizon=forecast_horizon,
        retrain=False,
        last_points_only=True
    )

    print("Extracting predictions specifically for June period...")
    # Extract True values for June only
    true_june = target_series.slice(eval_start, eval_end)
    
    # Intersect the time index to compare actuals vs predictions cleanly
    common_dates = true_june.time_index.intersection(forecasts.time_index)
    if len(common_dates) == 0:
        print("Error: No predictions generated for June. Need to check data availability bounding evaluation start.")
        return

    y_true = true_june.slice(common_dates[0], common_dates[-1]).values().flatten()
    y_pred = forecasts.slice(common_dates[0], common_dates[-1]).values().flatten()

    # Calculate Evaluation Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    sum_true = np.sum(np.abs(y_true))
    wape = np.sum(np.abs(y_true - y_pred)) / sum_true if sum_true != 0 else np.nan

    print("\n" + "="*40)
    print(f"Evaluation Results for June (web_id: {WEB_ID}):")
    print("="*40)
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"WAPE: {wape:.4f}")
    print("="*40 + "\n")

    # Plot Actual vs Predicted
    print("Generating Actual vs Predicted plot...")
    plt.figure(figsize=(12, 6))
    plt.plot(common_dates, y_true, label='Actual next_7d_sales', marker='o')
    plt.plot(common_dates, y_pred, label='Predicted next_7d_sales', linestyle='--', marker='x')
    plt.title(f'Actual vs Predicted Sales for {WEB_ID} (June Evaluation)')
    plt.xlabel('Date')
    plt.ylabel('Next 7d Sales')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_file = 'evaluation_performance_june.png'
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
    
    print(f"Plot correctly saved as '{plot_file}'")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    evaluate_june_data()
