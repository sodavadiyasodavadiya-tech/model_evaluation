import pandas as pd
import numpy as np
import torch
import json
import shap
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import TFTModel
import os
import streamlit_callback # Required for the PyTorch lightning checkpoint callback load 

# Ensure plots are saved without needing a display
import matplotlib
matplotlib.use('Agg')

class TFTWrapper(torch.nn.Module):
    def __init__(self, model, input_chunk_length, output_chunk_length, n_past, n_fut, n_stat):
        super().__init__()
        self.tft_inner = model.model # The PyTorch module
        self.icl = input_chunk_length
        self.ocl = output_chunk_length
        self.n_past = n_past
        self.n_fut = n_fut
        self.n_stat = n_stat
        
    def forward(self, x_flat):
        # x_flat: (batch, total_features)
        x_flat = x_flat.to(torch.float32)
        batch_size = x_flat.shape[0]
        
        # Split back into x_past, x_future, x_static
        # x_past: (batch, icl, n_past)
        # x_future: (batch, icl + ocl, n_fut)
        # x_static: (batch, n_stat)
        
        past_end = self.icl * self.n_past
        fut_end = past_end + (self.icl + self.ocl) * self.n_fut
        
        x_past = x_flat[:, :past_end].reshape(batch_size, self.icl, self.n_past)
        x_fut = x_flat[:, past_end:fut_end].reshape(batch_size, self.icl + self.ocl, self.n_fut)
        x_static = x_flat[:, fut_end:].reshape(batch_size, 1, self.n_stat)
        
        # Model forward pass
        # TFT typically expects a tuple (x_past, x_fut, x_static)
        out = self.tft_inner((x_past, x_fut, x_static))
        
        # We want to explain the point prediction (usually the middle quantile or the mean)
        # Ifquantiles are used, select the 0.5 quantile.
        # out shape: (batch, ocl, n_targets, n_quantiles)
        # Let's assume n_targets=1 and n_quantiles=1 for simplicity if not multi-quantile, 
        # but TFT is usually multi-quantile.
        if len(out.shape) == 4:
            # (batch, ocl, n_targets, n_quantiles) -> aggregate over ocl if needed or pick a step
            out = out[:, :, 0, out.shape[3]//2] # Middle quantile
            out = out.mean(dim=1) # Mean across 7 days
        else:
            # (batch, ocl, n_targets)
            out = out[:, :, 0].mean(dim=1)
            
        return out.unsqueeze(1) # Ensure (batch, 1) shape for SHAP

def get_base_feature_name(name):
    # e.g. "abd_carts_t-22" -> "abd_carts"
    # e.g. "clicks_fut_t0" -> "clicks"
    # e.g. "inventory_static" -> "inventory"
    base_name = name
    if "_t" in name:
        base_name = name.split("_t")[0]
    elif "_static" in name:
        base_name = name.split("_static")[0]
    
    if base_name.endswith("_fut"):
        base_name = base_name[:-4]
    return base_name

def generate_shap_june():
    DATA_PATH = 'prepared_data_with_target.csv'
    CONFIG_PATH = 'tft_model_4_month/training_config.json'
    MODEL_NAME = 'tft_model_4_month'
    WORK_DIR = '.'

    print("Loading configuration...")
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    selected_products = config.get('selected_products', [])
    if not selected_products:
        print("Error: No products in config.")
        return

    print(f"Loading data from {DATA_PATH}...")
    df_all = pd.read_csv(DATA_PATH)
    df_all['date'] = pd.to_datetime(df_all['date'])

    # Constants from config
    icl = config['model_parameters']['input_chunk_length']
    ocl = config['model_parameters']['output_chunk_length']
    target_col = config['target_column']
    past_covs_cols = config.get('past_covariates', [])
    fut_covs_cols = config.get('future_covariates', [])
    stat_covs_cols = config.get('static_covariates', [])
    all_numeric_cols = [target_col] + past_covs_cols + fut_covs_cols + stat_covs_cols

    eval_start = pd.Timestamp('2025-06-01')
    eval_end = pd.Timestamp('2025-06-23')

    inputs_list = []
    
    print(f"Building feature windows for all products ({len(selected_products)}) across June 2025...")
    for web_id in selected_products:
        df_web = df_all[df_all['web_id'] == web_id].copy().sort_values('date')
        if df_web.empty: continue

        for col in all_numeric_cols:
            if col in df_web.columns:
                df_web[col] = pd.to_numeric(df_web[col], errors='coerce').fillna(0)
            else:
                df_web[col] = 0

        try:
            target_ts = TimeSeries.from_dataframe(df_web, time_col='date', value_cols=target_col, static_covariates=df_web[stat_covs_cols].iloc[[0]])
            past_ts = TimeSeries.from_dataframe(df_web, time_col='date', value_cols=past_covs_cols) if past_covs_cols else None
            fut_ts = TimeSeries.from_dataframe(df_web, time_col='date', value_cols=fut_covs_cols) if fut_covs_cols else None
            
            # Find valid indices for June
            june_dates = df_web[(df_web['date'] >= eval_start) & (df_web['date'] <= eval_end)]
            if june_dates.empty: continue
            
            for d in june_dates['date']:
                idx = df_web[df_web['date'] == d].index[0]
                start_idx = df_web.index.get_loc(idx)

                if start_idx < icl or (start_idx + ocl) > len(df_web):
                    continue

                p_target = target_ts.values()[start_idx - icl : start_idx]
                p_past = past_ts.values()[start_idx - icl : start_idx] if past_ts else np.zeros((icl, 0))
                p_fut = fut_ts.values()[start_idx - icl : start_idx] if fut_ts else np.zeros((icl, 0))
                
                x_p = np.concatenate([p_target, p_past, p_fut], axis=1)
                f_fut = fut_ts.values()[start_idx - icl : start_idx + ocl] if fut_ts else np.zeros((icl + ocl, 0))
                x_f = f_fut
                x_s = target_ts.static_covariates_values().flatten()
                
                flat = np.concatenate([x_p.flatten(), x_f.flatten(), x_s.flatten()])
                inputs_list.append(flat)
        except:
            continue

    X_total = np.array(inputs_list)
    if X_total.shape[0] == 0:
        print("Error: No valid windows found.")
        return
        
    print(f"Total windows built: {X_total.shape[0]}")

    # Sampling Logic
    num_to_explain = min(20, len(X_total))
    explain_indices = np.random.choice(len(X_total), num_to_explain, replace=False)
    X_explain = X_total[explain_indices]

    bg_size = min(5, len(X_total))
    bg_indices = np.random.choice(len(X_total), bg_size, replace=False)
    X_background = X_total[bg_indices]

    # Load Model
    print("Loading TFT model...")
    model = TFTModel.load_from_checkpoint(model_name=MODEL_NAME, work_dir=WORK_DIR, best=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_past_feat = 1 + len(past_covs_cols) + len(fut_covs_cols)
    n_fut_feat = len(fut_covs_cols)
    n_stat_feat = len(stat_covs_cols)
    
    wrapper = TFTWrapper(model, icl, ocl, n_past_feat, n_fut_feat, n_stat_feat)
    wrapper.to(device).float().eval()

    background_tensor = torch.tensor(X_background, dtype=torch.float32).to(device)
    explain_tensor = torch.tensor(X_explain, dtype=torch.float32).to(device)

    print(f"Computing SHAP values for {num_to_explain} samples using background of size {bg_size}...")
    explainer = shap.GradientExplainer(wrapper, background_tensor)
    shap_results = explainer.shap_values(explain_tensor)
    
    if isinstance(shap_results, list):
        shap_values = np.array(shap_results[0])
    else:
        shap_values = np.array(shap_results)
        
    if len(shap_values.shape) == 3 and shap_values.shape[2] == 1:
        shap_values = shap_values[:, :, 0]
        
    # Feature Names Construction
    feat_names = []
    # Past (Lagged)
    for t in range(-icl, 0):
        feat_names.append(f"{target_col}_t{t}")
        for c in past_covs_cols: feat_names.append(f"{c}_t{t}")
        for c in fut_covs_cols: feat_names.append(f"{c}_t{t}")
    # Future
    for t in range(-icl, ocl):
        for c in fut_covs_cols: feat_names.append(f"{c}_fut_t{t}")
    # Static
    for c in stat_covs_cols:
        feat_names.append(f"{c}_static")

    # Group by Base Feature
    print("Aggregating SHAP values by base feature name...")
    aggregated_importance = {}
    for i, name in enumerate(feat_names):
        # Exclude target from importance if desired, but here we'll include all features including lags
        base_name = get_base_feature_name(name)
        if base_name not in aggregated_importance:
            aggregated_importance[base_name] = np.zeros(shap_values.shape[0])
        aggregated_importance[base_name] += shap_values[:, i]

    final_importance = []
    for base_name, samples_shap in aggregated_importance.items():
        if base_name == target_col: continue # Filter out target if requested, but let's keep all features for visibility
        final_importance.append({
            "feature": base_name,
            "importance": np.abs(samples_shap).mean()
        })

    importance_df = pd.DataFrame(final_importance).sort_values("importance", ascending=False)
    top_20 = importance_df.head(20)

    # Visualization
    plt.figure(figsize=(12, 8))
    plt.barh(top_20["feature"], top_20["importance"], color='lightgreen', edgecolor='green')
    plt.gca().invert_yaxis()
    plt.title("Collective Global SHAP Feature Importance (June 2025 - All Products)")
    plt.xlabel("Mean |SHAP value|")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("shap_global_importance_june_collective.png")
    plt.close()

    print("Collective SHAP analysis completed.")

    # -------------------------------------------------------------
    # PART 2: Local SHAP Explanation for specific product and date
    # -------------------------------------------------------------
    target_web_id = 'qb1146021'
    target_date = pd.Timestamp('2025-06-01')
    print(f"\nGenerating local SHAP explanation for {target_web_id} on {target_date.date()}...")

    df_local = df_all[df_all['web_id'] == target_web_id].copy().sort_values('date')
    if df_local.empty:
        print(f"Error: No data found for product {target_web_id}")
    else:
        # Fill missing numeric columns
        for col in all_numeric_cols:
            if col in df_local.columns:
                df_local[col] = pd.to_numeric(df_local[col], errors='coerce').fillna(0)
            else:
                df_local[col] = 0

        try:
            # Prepare Series for the product
            l_target_ts = TimeSeries.from_dataframe(df_local, time_col='date', value_cols=target_col, static_covariates=df_local[stat_covs_cols].iloc[[0]])
            l_past_ts = TimeSeries.from_dataframe(df_local, time_col='date', value_cols=past_covs_cols) if past_covs_cols else None
            l_fut_ts = TimeSeries.from_dataframe(df_local, time_col='date', value_cols=fut_covs_cols) if fut_covs_cols else None
            
            # Find the specific row for June 1
            mask = df_local['date'] == target_date
            if not mask.any():
                print(f"Error: Date {target_date.date()} not found for product {target_web_id}")
            else:
                target_row_idx = df_local.index[mask][0]
                start_loc = df_local.index.get_loc(target_row_idx)

                if start_loc < icl or (start_loc + ocl) > len(df_local):
                    print(f"Error: Insufficient history or future data for {target_date.date()} window.")
                else:
                    # Extract window features exactly as in the global building loop
                    p_target = l_target_ts.values()[start_loc - icl : start_loc]
                    p_past = l_past_ts.values()[start_loc - icl : start_loc] if l_past_ts else np.zeros((icl, 0))
                    p_fut = l_fut_ts.values()[start_loc - icl : start_loc] if l_fut_ts else np.zeros((icl, 0))
                    
                    x_p = np.concatenate([p_target, p_past, p_fut], axis=1)
                    f_fut = l_fut_ts.values()[start_loc - icl : start_loc + ocl] if l_fut_ts else np.zeros((icl + ocl, 0))
                    x_f = f_fut
                    x_s = l_target_ts.static_covariates_values().flatten()
                    
                    x_local_flat = np.concatenate([x_p.flatten(), x_f.flatten(), x_s.flatten()])
                    x_local_tensor = torch.tensor([x_local_flat], dtype=torch.float32).to(device)

                    # Compute SHAP values for this single sample
                    shap_results_local = explainer.shap_values(x_local_tensor)
                    if isinstance(shap_results_local, list):
                        shap_val_single = np.array(shap_results_local[0][0])
                    else:
                        shap_val_single = np.array(shap_results_local[0])
                    
                    # Ensure 1D
                    if len(shap_val_single.shape) > 1:
                        shap_val_single = shap_val_single.flatten()
                    
                    # Group by base feature and remove target lags
                    local_agg_shap = {}
                    local_agg_data = {}
                    
                    for i, name in enumerate(feat_names):
                        base = get_base_feature_name(name)
                        if base == target_col: continue # Remove target lag features
                        
                        if base not in local_agg_shap:
                            local_agg_shap[base] = 0
                            local_agg_data[base] = []
                        local_agg_shap[base] += shap_val_single[i]
                        local_agg_data[base].append(x_local_flat[i])
                    
                    # Prepare data for Explanation object
                    final_local_features = list(local_agg_shap.keys())
                    final_local_shap = np.array([local_agg_shap[f] for f in final_local_features])
                    # Use mean of lag values as the representative 'data' value for the feature
                    final_local_data = np.array([np.mean(local_agg_data[f]) for f in final_local_features])
                    
                    # Get base value (expected value)
                    try:
                        base_val = explainer.expected_value
                        if isinstance(base_val, (list, np.ndarray)) and len(base_val) > 0:
                            base_val = base_val[0]
                    except AttributeError:
                        # Fallback: compute mean prediction on background
                        with torch.no_grad():
                            base_val = wrapper(background_tensor).mean().item()
                    
                    # Force base_val to be scalar if it's an array
                    if hasattr(base_val, "item"):
                        base_val = base_val.item()
                    elif isinstance(base_val, (list, np.ndarray)):
                        base_val = np.array(base_val).flatten()[0]

                    # Create SHAP Explanation object
                    exp = shap.Explanation(
                        values=final_local_shap,
                        base_values=base_val,
                        data=final_local_data,
                        feature_names=final_local_features
                    )

                    # Generate and save Waterfall Plot
                    plt.figure(figsize=(12, 10))
                    shap.plots.waterfall(exp, max_display=20, show=False)
                    plt.title("Local SHAP Explanation – Product qb1146021 (June 1 Prediction)")
                    plt.tight_layout()
                    plt.savefig("shap_local_qb1146021_june1.png")
                    plt.close()
                    print("Local SHAP waterfall plot saved as shap_local_qb1146021_june1.png")

        except Exception as e:
            print(f"Error during local explanation: {e}")

if __name__ == "__main__":
    generate_shap_june()
