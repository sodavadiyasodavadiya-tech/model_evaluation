
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from darts import TimeSeries
from darts.models import TFTModel
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Ensure plots are saved without needing a display
import matplotlib
matplotlib.use('Agg')

def load_data(data_path, config_path):
    """
    Load the dataset and training configuration.
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return df, config

def prepare_timeseries(df, config, products):
    """
    Convert pandas DataFrame into Darts TimeSeries objects for each product.
    """
    target_col = config['target_column']
    past_covs_cols = config.get('past_covariates', [])
    fut_covs_cols = config.get('future_covariates', [])
    stat_covs_cols = config.get('static_covariates', [])
    
    target_list = []
    past_list = []
    fut_list = []
    valid_products = []
    
    print(f"Preparing TimeSeries for {len(products)} products...")
    for web_id in products:
        df_web = df[df['web_id'] == web_id].copy().sort_values('date')
        if df_web.empty:
            continue
            
        # Ensure numeric types
        for col in [target_col] + past_covs_cols + fut_covs_cols + stat_covs_cols:
            if col in df_web.columns:
                df_web[col] = pd.to_numeric(df_web[col], errors='coerce').fillna(0)
        
        try:
            # Extract static covariates for this specific product
            stat_covs = df_web[stat_covs_cols].iloc[[0]] if stat_covs_cols else None
            
            ts_target = TimeSeries.from_dataframe(df_web, time_col='date', value_cols=target_col, static_covariates=stat_covs)
            
            ts_past = None
            if past_covs_cols:
                ts_past = TimeSeries.from_dataframe(df_web, time_col='date', value_cols=past_covs_cols)
            
            ts_fut = None
            if fut_covs_cols:
                ts_fut = TimeSeries.from_dataframe(df_web, time_col='date', value_cols=fut_covs_cols)
                
            target_list.append(ts_target)
            past_list.append(ts_past)
            fut_list.append(ts_fut)
            valid_products.append(web_id)
        except Exception as e:
            # Product might have too little data or frequency issues
            continue
            
    return target_list, past_list, fut_list, valid_products

def run_forecast(model_name, work_dir, target_series, past_series, future_series, start_date):
    """
    Run rolling forecasts using the trained TFT model.
    """
    print(f"Loading TFT model '{model_name}'...")
    model = TFTModel.load_from_checkpoint(model_name=model_name, work_dir=work_dir, best=True)
    
    print(f"Generating rolling forecasts for {len(target_series)} products (Forecast Horizon = 7)...")
    forecasts = model.historical_forecasts(
        series=target_series,
        past_covariates=past_series if any(s is not None for s in past_series) else None,
        future_covariates=future_series if any(s is not None for s in future_series) else None,
        start=start_date,
        forecast_horizon=7,
        retrain=False,
        last_points_only=True,
        verbose=True
    )
    
    if not isinstance(forecasts, list):
        forecasts = [forecasts]
        
    return forecasts

def compute_metrics(target_series, forecasts, start_date, end_date, product_ids):
    """
    Compute per-product and collective (micro/macro) metrics.
    """
    print("Computing metrics and aligning results...")
    results = []
    all_y_true = []
    all_y_pred = []
    
    for i, product_id in enumerate(product_ids):
        # Extract the target series window for evaluation
        ts_actual = target_series[i].slice(start_date, end_date)
        ts_forecast = forecasts[i]
        
        # Align time indices
        common_dates = ts_actual.time_index.intersection(ts_forecast.time_index)
        if len(common_dates) == 0:
            continue
            
        y_true = ts_actual.slice(common_dates[0], common_dates[-1]).values().flatten()
        y_pred = ts_forecast.slice(common_dates[0], common_dates[-1]).values().flatten()
        
        # Per-product metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        sum_true = np.sum(np.abs(y_true))
        wape = np.sum(np.abs(y_true - y_pred)) / sum_true if sum_true != 0 else np.nan
        
        results.append({
            'web_id': product_id,
            'mae': mae,
            'rmse': rmse,
            'wape': wape,
            'y_true': y_true,
            'y_pred': y_pred
        })
        
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        
    # Micro Average Metrics (Global weighted performance)
    y_true_all = np.array(all_y_true)
    y_pred_all = np.array(all_y_pred)
    
    micro_mae = mean_absolute_error(y_true_all, y_pred_all)
    micro_rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    sum_true_all = np.sum(np.abs(y_true_all))
    micro_wape = np.sum(np.abs(y_true_all - y_pred_all)) / sum_true_all if sum_true_all != 0 else np.nan
    
    # Macro Metrics (Simple average across product scores)
    macro_mae = np.mean([r['mae'] for r in results])
    macro_rmse = np.mean([r['rmse'] for r in results])
    
    collective_metrics = {
        'micro_mae': micro_mae,
        'micro_rmse': micro_rmse,
        'micro_wape': micro_wape,
        'macro_mae': macro_mae,
        'macro_rmse': macro_rmse
    }
    
    return results, collective_metrics, y_true_all, y_pred_all

def save_results(per_product_results, collective_metrics, csv_path, txt_path):
    """
    Save evaluation metrics to disk.
    """
    # Save per-product CSV
    df_per_product = pd.DataFrame(per_product_results).drop(columns=['y_true', 'y_pred'])
    df_per_product.to_csv(csv_path, index=False)
    print(f"Per-product metrics saved to: {csv_path}")
    
    # Save collective TXT
    with open(txt_path, 'w') as f:
        f.write("Collective Evaluation Metrics (June 2025)\n")
        f.write("="*50 + "\n")
        f.write(f"Period: June 1st to June 30th\n")
        f.write(f"Number of products evaluated: {len(per_product_results)}\n")
        f.write("="*50 + "\n\n")
        f.write("[MICRO METRICS - Global Average]\n")
        f.write(f"Micro MAE:   {collective_metrics['micro_mae']:.4f}\n")
        f.write(f"Micro RMSE:  {collective_metrics['micro_rmse']:.4f}\n")
        f.write(f"Micro WAPE:  {collective_metrics['micro_wape']:.4f}\n\n")
        f.write("[MACRO METRICS - Average Across Products]\n")
        f.write(f"Macro MAE:   {collective_metrics['macro_mae']:.4f}\n")
        f.write(f"Macro RMSE:  {collective_metrics['macro_rmse']:.4f}\n\n")
        f.write("="*50 + "\n")
    print(f"Collective metrics saved to: {txt_path}")

def create_plots(y_true, y_pred):
    """
    Generate diagnostic visualizations.
    """
    print("Generating evaluation plots...")
    
    # 1. Scatter Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.3, color='blue', s=10)
    m_val = max(y_true.max(), y_pred.max()) if len(y_true) > 0 else 1
    plt.plot([0, m_val], [0, m_val], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Sales (Units)')
    plt.ylabel('Predicted Sales (Units)')
    plt.title('Collective Actual vs Predicted (June Evaluation)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('actual_vs_predicted_combined.png', dpi=300)
    plt.close()
    
    # 2. Error Histogram
    errors = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel('Prediction Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True, alpha=0.3)
    plt.savefig('error_distribution_histogram.png', dpi=300)
    plt.close()
    
    print("Plots saved: actual_vs_predicted_combined.png, error_distribution_histogram.png")

def main():
    warnings.filterwarnings("ignore")
    
    # --- Paths ---
    DATA_PATH = 'prepared_data_with_target.csv'
    CONFIG_PATH = 'tft_model_4_month/training_config.json'
    MODEL_NAME = 'tft_model_4_month'
    WORK_DIR = '.'
    
    # --- Dates ---
    EVAL_START = pd.Timestamp('2025-06-01')
    EVAL_END = pd.Timestamp('2025-06-30')
    
    try:
        # 1. Load Data
        df, config = load_data(DATA_PATH, CONFIG_PATH)
        selected_products = config.get('selected_products', [])
        
        # 2. Prepare Data for Darts
        ts_targets, ts_pasts, ts_futs, valid_web_ids = prepare_timeseries(df, config, selected_products)
        
        # 3. Execution (Forecasting)
        forecasts = run_forecast(MODEL_NAME, WORK_DIR, ts_targets, ts_pasts, ts_futs, EVAL_START)
        
        # 4. Computation (Metrics)
        per_product_results, collective_metrics, y_true_all, y_pred_all = compute_metrics(
            ts_targets, forecasts, EVAL_START, EVAL_END, valid_web_ids
        )
        
        # 5. Export
        save_results(per_product_results, collective_metrics, 'per_product_metrics.csv', 'collective_metrics.txt')
        
        # 6. Visualization
        create_plots(y_true_all, y_pred_all)
        
        print("\n" + "="*50)
        print("EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
        print("="*50)
        
    except Exception as e:
        print(f"\nCRITICAL ERROR in pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
