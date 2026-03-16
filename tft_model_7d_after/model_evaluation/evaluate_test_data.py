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

def load_data(test_data_path, config_path):
    """
    Load the test dataset and training configuration.
    """
    print(f"Loading test data from {test_data_path}...")
    df = pd.read_csv(test_data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return df, config

def prepare_timeseries(df, config, products):
    """
    Convert pandas DataFrame into Darts TimeSeries objects for test evaluation.
    """
    target_col = config['target_column']
    past_covs_cols = config.get('past_covariates', [])
    fut_covs_cols = config.get('future_covariates', [])
    stat_covs_cols = config.get('static_covariates', [])
    
    target_list = []
    past_list = []
    fut_list = []
    valid_products = []
    
    print(f"Preparing TimeSeries for {len(products)} products from test data...")
    for web_id in products:
        df_web = df[df['web_id'] == web_id].copy().sort_values('date')
        if df_web.empty:
            continue
            
        # Ensure numeric types and handle missing columns in test data
        for col in [target_col] + past_covs_cols + fut_covs_cols + stat_covs_cols:
            if col in df_web.columns:
                df_web[col] = pd.to_numeric(df_web[col], errors='coerce').fillna(0)
            else:
                df_web[col] = 0
        
        try:
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
        except:
            continue
            
    return target_list, past_list, fut_list, valid_products

def run_forecast(model_dir, work_dir, checkpoint_file, target_series, past_series, future_series, ocl):
    """
    Run rolling forecasts using the trained TFT model for test evaluation.
    """
    print(f"Loading TFT model from {model_dir}...")
    model = TFTModel.load_from_checkpoint(model_name=model_dir, work_dir=work_dir, file_name=checkpoint_file)
    model.model.eval()
    
    print(f"Generating rolling forecasts for {len(target_series)} products (Forecast Horizon = {ocl})...")
    forecasts = model.historical_forecasts(
        series=target_series,
        past_covariates=past_series if any(s is not None for s in past_series) else None,
        future_covariates=future_series if any(s is not None for s in future_series) else None,
        start=None, # Start at earliest valid point
        forecast_horizon=ocl,
        retrain=False,
        last_points_only=True,
        verbose=True
    )
    
    if not isinstance(forecasts, list):
        forecasts = [forecasts]
        
    return forecasts

def compute_metrics(target_series, forecasts, product_ids):
    """
    Compute alignment and metrics for test evaluation.
    """
    print("Computing metrics and aligning results...")
    results = []
    all_y_true = []
    all_y_pred = []
    
    for i, product_id in enumerate(product_ids):
        ts_actual = target_series[i]
        ts_forecast = forecasts[i]
        
        if ts_forecast is None:
            continue
            
        common_dates = ts_actual.time_index.intersection(ts_forecast.time_index)
        if len(common_dates) == 0:
            continue
            
        y_true = ts_actual.slice(common_dates[0], common_dates[-1]).values().flatten()
        y_pred = ts_forecast.slice(common_dates[0], common_dates[-1]).values().flatten()
        
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
        
    y_true_all = np.array(all_y_true)
    y_pred_all = np.array(all_y_pred)
    
    micro_mae = mean_absolute_error(y_true_all, y_pred_all)
    micro_rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    sum_true_all = np.sum(np.abs(y_true_all))
    micro_wape = np.sum(np.abs(y_true_all - y_pred_all)) / sum_true_all if sum_true_all != 0 else np.nan
    
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
    Save evaluation metrics for test dataset.
    """
    df_per_product = pd.DataFrame(per_product_results).drop(columns=['y_true', 'y_pred'])
    df_per_product.to_csv(csv_path, index=False)
    print(f"Per-product test metrics saved to: {csv_path}")
    
    with open(txt_path, 'w') as f:
        f.write("Collective Evaluation Metrics (Test Dataset)\n")
        f.write("="*50 + "\n")
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
    print(f"Collective test metrics saved to: {txt_path}")

def create_plots(y_true, y_pred):
    """
    Generate visualizations for test evaluation.
    """
    print("Generating test evaluation plots...")
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.3, color='purple', s=10)
    m_val = max(y_true.max(), y_pred.max()) if len(y_true) > 0 else 1
    plt.plot([0, m_val], [0, m_val], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Sales (Units)')
    plt.ylabel('Predicted Sales (Units)')
    plt.title('Collective Test Data Performance (Actual vs Predicted)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('actual_vs_predicted_test.png', dpi=300)
    plt.close()
    
    errors = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, color='plum', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel('Prediction Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Test Data Prediction Errors')
    plt.grid(True, alpha=0.3)
    plt.savefig('error_distribution_test_histogram.png', dpi=300)
    plt.close()
    
    print("Plots saved: actual_vs_predicted_test.png, error_distribution_test_histogram.png")

def main():
    warnings.filterwarnings("ignore")
    
    # --- Paths ---
    TEST_DATA_PATH = 'test_data.csv'
    CONFIG_PATH = 'tft_model_after_7d/training_config.json'
    MODEL_DIR = 'tft_model_after_7d'
    CHECKPOINT_FILE = 'best-epoch=8-val_loss=1.04.ckpt' 
    WORK_DIR = '.'
    
    try:
        # 1. Load
        df, config = load_data(TEST_DATA_PATH, CONFIG_PATH)
        selected_products = config.get('selected_products', [])
        ocl = config['model_parameters']['output_chunk_length']
        
        # 2. Prepare
        ts_targets, ts_pasts, ts_futs, valid_web_ids = prepare_timeseries(df, config, selected_products)
        
        # 3. Forecast
        forecasts = run_forecast(MODEL_DIR, WORK_DIR, CHECKPOINT_FILE, ts_targets, ts_pasts, ts_futs, ocl)
        
        # 4. Compute
        per_product_results, collective_metrics, y_true_all, y_pred_all = compute_metrics(
            ts_targets, forecasts, valid_web_ids
        )
        
        # 5. Save
        save_results(per_product_results, collective_metrics, 'per_product_metrics_test.csv', 'collective_metrics_test.txt')
        
        # 6. Plot
        create_plots(y_true_all, y_pred_all)
        
        print("\n" + "="*50)
        print("TEST DATA EVALUATION PIPELINE COMPLETED")
        print("="*50)
        
    except Exception as e:
        print(f"\nCRITICAL ERROR in pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
