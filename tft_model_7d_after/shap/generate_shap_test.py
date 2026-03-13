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
        
        past_end = self.icl * self.n_past
        fut_end = past_end + (self.icl + self.ocl) * self.n_fut
        
        x_past = x_flat[:, :past_end].reshape(batch_size, self.icl, self.n_past)
        x_fut = x_flat[:, past_end:fut_end].reshape(batch_size, self.icl + self.ocl, self.n_fut)
        x_static = x_flat[:, fut_end:].reshape(batch_size, 1, self.n_stat)
        
        # Model forward pass
        out = self.tft_inner((x_past, x_fut, x_static))
        
        if len(out.shape) == 4:
            # (batch, ocl, n_targets, n_quantiles)
            out = out[:, :, 0, out.shape[3]//2] # Middle quantile
            out = out.mean(dim=1) # Mean across the horizon (default behavior)
        else:
            # (batch, ocl, n_targets)
            out = out[:, :, 0].mean(dim=1)
            
        return out.unsqueeze(1) # Ensure (batch, 1) shape for SHAP

def get_base_feature_name(name):
    base_name = name
    if "_t" in name:
        base_name = name.split("_t")[0]
    elif "_static" in name:
        base_name = name.split("_static")[0]
    
    if base_name.endswith("_fut"):
        base_name = base_name[:-4]
    return base_name

def generate_shap_test():
    WEB_ID = 'qb1146021'
    DATA_PATH = 'test_data.csv'
    TRAIN_DATA_PATH = 'prepared_data_with_target.csv' # For extra history if needed
    CONFIG_PATH = 'tft_model_after_7d/training_config.json'
    MODEL_NAME = 'tft_model_after_7d'
    WORK_DIR = '.'

    print(f"Loading configuration from {CONFIG_PATH}...")
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    if 'selected_products' in config and len(config['selected_products']) > 0:
        # We might want to stick to the same product as previous evaluations
        # But if not specified, take from config
        pass

    print(f"Loading test data for {WEB_ID}...")
    df_test = pd.read_csv(DATA_PATH)
    df_test['date'] = pd.to_datetime(df_test['date'])
    df_web = df_test[df_test['web_id'] == WEB_ID].copy().sort_values('date')

    if df_web.empty:
        print(f"Error: Product {WEB_ID} not found in {DATA_PATH}")
        return

    # Constants from config
    icl = config['model_parameters']['input_chunk_length']
    ocl = config['model_parameters']['output_chunk_length']
    target_col = config['target_column']
    past_covs_cols = config.get('past_covariates', [])
    fut_covs_cols = config.get('future_covariates', [])
    stat_covs_cols = config.get('static_covariates', [])

    # Numeric conversion
    all_cols = [target_col] + past_covs_cols + fut_covs_cols + stat_covs_cols
    for col in all_cols:
        if col in df_web.columns:
            df_web[col] = pd.to_numeric(df_web[col], errors='coerce').fillna(0)

    # TimeSeries
    # Static covariates from the first row
    stat_covs_df = df_web[stat_covs_cols].iloc[[0]] if stat_covs_cols else None
    
    target_ts = TimeSeries.from_dataframe(df_web, time_col='date', value_cols=target_col, static_covariates=stat_covs_df)
    past_ts = TimeSeries.from_dataframe(df_web, time_col='date', value_cols=past_covs_cols) if past_covs_cols else None
    fut_ts = TimeSeries.from_dataframe(df_web, time_col='date', value_cols=fut_covs_cols) if fut_covs_cols else None

    # Load Model
    print(f"Loading TFT model {MODEL_NAME}...")
    model = TFTModel.load_from_checkpoint(model_name=MODEL_NAME, work_dir=WORK_DIR, best=True)
    
    # Prepare Inputs for SHAP
    test_dates = df_web['date'].unique()
    
    inputs_list = []
    dates_list = []
    
    n_past_feat = 1 + len(past_covs_cols) + len(fut_covs_cols)
    n_fut_feat = len(fut_covs_cols)
    n_stat_feat = len(stat_covs_cols)
    
    print("Constructing feature windows for SHAP from test data...")
    for d in test_dates:
        try:
            target_val_at_d = df_web[df_web['date'] == d]
            if target_val_at_d.empty: continue
            start_idx = df_web.index.get_loc(target_val_at_d.index[0])
            
            if start_idx < icl: continue
            
            p_target = target_ts.values()[start_idx - icl : start_idx]
            p_past = past_ts.values()[start_idx - icl : start_idx] if past_ts else np.zeros((icl, 0))
            p_fut = fut_ts.values()[start_idx - icl : start_idx] if fut_ts else np.zeros((icl, 0))
            
            if len(p_target) != icl or len(p_past) != icl or len(p_fut) != icl:
                continue

            x_p = np.concatenate([p_target, p_past, p_fut], axis=1) # (icl, n_past)
            
            f_fut = fut_ts.values()[start_idx - icl : start_idx + ocl] if fut_ts else np.zeros((icl + ocl, 0))
            if len(f_fut) != (icl + ocl):
                continue
            
            x_f = f_fut # (icl + ocl, n_fut)
            x_s = target_ts.static_covariates_values().flatten() # (n_stat)
            
            flat = np.concatenate([x_p.flatten(), x_f.flatten(), x_s.flatten()])
            inputs_list.append(flat)
            dates_list.append(d)
        except Exception as e:
            continue

    X_test = np.array(inputs_list)
    Y_dates = np.array(dates_list)
    if X_test.shape[0] == 0:
        print("Error: No valid test windows found. Check if test data length is sufficient.")
        return
        
    print(f"Total valid windows for SHAP: {X_test.shape[0]}")

    # Device Handling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    wrapper = TFTWrapper(model, icl, ocl, n_past_feat, n_fut_feat, n_stat_feat)
    wrapper.to(device)
    wrapper.float()
    wrapper.eval()

    # Sampling
    # we take the very last window for local explanation, and random 19 others for global
    last_idx = len(X_test) - 1
    other_indices = np.random.choice(last_idx, min(19, last_idx), replace=False)
    
    # Put last_idx first so filtered_shap[0] is the last day
    indices = np.concatenate([[last_idx], other_indices])
    explain_data = torch.tensor(X_test[indices], dtype=torch.float32).to(device)
    explain_dates = Y_dates[indices]

    num_samples = len(indices)
    bg_size = min(10, len(X_test))
    bg_indices = np.random.choice(len(X_test), bg_size, replace=False)
    background = torch.tensor(X_test[bg_indices], dtype=torch.float32).to(device)

    print("Initializing GradientExplainer...")
    explainer = shap.GradientExplainer(wrapper, background)
    
    print(f"Computing SHAP values for {num_samples} samples...")
    shap_results = explainer.shap_values(explain_data)
    
    if isinstance(shap_results, list):
        shap_values = np.array(shap_results[0])
    else:
        shap_values = np.array(shap_results)
        
    if len(shap_values.shape) == 3 and shap_values.shape[2] == 1:
        shap_values = shap_values[:, :, 0]
        
    # Feature Names
    feat_names = []
    # Past
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

    # Exclude target
    include_indices = [i for i, name in enumerate(feat_names) if target_col not in name]
    filtered_shap = shap_values[:, include_indices]
    filtered_names = [feat_names[i] for i in include_indices]

    # Global Aggregation
    print("Aggregating SHAP values by base feature...")
    aggregated_importance = {}
    for i, name in enumerate(filtered_names):
        base_name = get_base_feature_name(name)
        if base_name not in aggregated_importance:
            aggregated_importance[base_name] = np.zeros(filtered_shap.shape[0])
        aggregated_importance[base_name] += filtered_shap[:, i]

    feature_importance_list = []
    for base_name, sample_shaps in aggregated_importance.items():
        mean_abs_importance = np.abs(sample_shaps).mean()
        if mean_abs_importance > 1e-10:
            feature_importance_list.append({"feature": base_name, "importance": mean_abs_importance})

    importance_df = pd.DataFrame(feature_importance_list)
    if not importance_df.empty:
        importance_df = importance_df.sort_values("importance", ascending=False)
        top_features = importance_df.head(20)

        print("Generating Global Aggregated SHAP Importance plot...")
        plt.figure(figsize=(12, 8))
        plt.barh(top_features["feature"], top_features["importance"], color='lightcoral', edgecolor='brown')
        plt.gca().invert_yaxis()
        plt.title(f"Global SHAP Feature Importance (Test Data - {WEB_ID})")
        plt.xlabel("Mean |SHAP value|")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("shap_global_importance_test.png")
        plt.close()

    # Local Aggregation (for the last sample)
    explained_date = pd.to_datetime(explain_dates[0]).strftime('%Y-%m-%d')
    print(f"Generating Local Explanation plot for date: {explained_date}...")
    plt.figure(figsize=(16, 12))
    
    base_val = getattr(explainer, 'expected_value', None)
    if base_val is None:
        with torch.no_grad():
            base_val = wrapper(background).mean().item()
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[0]
        
    local_aggregated = {}
    for i, name in enumerate(filtered_names):
        base_name = get_base_feature_name(name)
        if base_name not in local_aggregated:
            local_aggregated[base_name] = 0
        local_aggregated[base_name] += filtered_shap[0, i]
        
    local_names = list(local_aggregated.keys())
    local_vals = np.array(list(local_aggregated.values()))
    
    shap.plots._waterfall.waterfall_legacy(base_val, local_vals, feature_names=local_names, max_display=20, show=False)
    plt.title(f"Top 20 Local Impact Features (Aggregated) - Test Data ({explained_date})")
    plt.savefig("shap_local_impact_test.png", bbox_inches='tight')
    plt.close()

    print("SHAP explainability for test data completed.")
    print("Files: shap_global_importance_test.png, shap_local_impact_test.png")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    generate_shap_test()
