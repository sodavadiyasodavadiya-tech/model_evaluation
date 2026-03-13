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
    WEB_ID = 'qb1146021'
    DATA_PATH = 'prepared_data_with_target.csv'
    CONFIG_PATH = 'tft_model_4_month/training_config.json'
    MODEL_NAME = 'tft_model_4_month'
    WORK_DIR = '.'

    print("Loading configuration...")
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    if 'selected_products' in config and len(config['selected_products']) > 0:
        WEB_ID = config['selected_products'][0]

    print(f"Loading data for {WEB_ID}...")
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df_web = df[df['web_id'] == WEB_ID].copy().sort_values('date')

    # Constants from config
    icl = config['model_parameters']['input_chunk_length']
    ocl = config['model_parameters']['output_chunk_length']
    target_col = config['target_column']
    past_covs_cols = config.get('past_covariates', [])
    fut_covs_cols = config.get('future_covariates', [])
    stat_covs_cols = config.get('static_covariates', [])

    # Prepare June Data
    eval_start = pd.Timestamp('2025-06-01')
    eval_end = pd.Timestamp('2025-06-30')
    
    # We need some history before June 1st to have a full sliding window
    # But SHAP explanations are ONLY for June predictions
    # Predictions starting from June 1st need data from June 1st - icl
    df_eval = df_web[df_web['date'] <= eval_end].copy()

    # Numeric conversion
    all_cols = [target_col] + past_covs_cols + fut_covs_cols + stat_covs_cols
    for col in all_cols:
        df_web[col] = pd.to_numeric(df_web[col], errors='coerce').fillna(0)

    # Prepare June Data with enough buffer for future covariates
    # We need history before June 1st and also ocl days after the prediction point in June
    df_eval = df_web[df_web['date'] <= eval_end + pd.Timedelta(days=ocl)].copy()

    # TimeSeries
    target_ts = TimeSeries.from_dataframe(df_eval, time_col='date', value_cols=target_col, static_covariates=df_eval[stat_covs_cols].iloc[[0]])
    past_ts = TimeSeries.from_dataframe(df_eval, time_col='date', value_cols=past_covs_cols) if past_covs_cols else None
    fut_ts = TimeSeries.from_dataframe(df_eval, time_col='date', value_cols=fut_covs_cols) if fut_covs_cols else None

    # Load Model
    print("Loading TFT model...")
    model = TFTModel.load_from_checkpoint(model_name=MODEL_NAME, work_dir=WORK_DIR, best=True)
    
    # Prepare Inputs for SHAP
    # We create inputs for June only. 
    # For each day in June, we have an input window.
    june_dates = df_eval[(df_eval['date'] >= eval_start) & (df_eval['date'] <= eval_end)]['date'].unique()
    
    inputs_list = []
    
    # Feature Names Construction
    # Order in x_past: Target, Past Covs, Future Covs (at past)
    # Order in x_fut: Future Covs
    # Order in x_stat: Static
    
    n_past_feat = 1 + len(past_covs_cols) + len(fut_covs_cols)
    n_fut_feat = len(fut_covs_cols)
    n_stat_feat = len(stat_covs_cols)
    
    print("Constructing feature windows for SHAP...")
    for d in june_dates:
        # Get target, past, future blocks
        # x_past: icl steps ending at d-1
        # x_fut: icl + ocl steps ending at d-1+ocl
        # This is the standard Darts way
        
        try:
            # Requesting input leading to d
            # Darts uses 'start' as the first prediction point
            # model._create_dataset will return a tuple (past_target, past_covs, historic_future_covs, future_covs, static_covs)
            val_series, val_past, val_fut = target_ts, past_ts, fut_ts
            
            # Manual extraction to ensure exact alignment
            # Past slice: [d - icl : d] (exclusive of d for past target)
            idx = df_eval[df_eval['date'] == d].index[0]
            start_idx = df_eval.index.get_loc(idx)
            
            p_target = target_ts.values()[start_idx - icl : start_idx]
            p_past = past_ts.values()[start_idx - icl : start_idx] if past_ts else np.zeros((icl, 0))
            p_fut = fut_ts.values()[start_idx - icl : start_idx] if fut_ts else np.zeros((icl, 0))
            
            # Check lengths
            if len(p_target) != icl or len(p_past) != icl or len(p_fut) != icl:
                continue

            x_p = np.concatenate([p_target, p_past, p_fut], axis=1) # (icl, n_past)
            
            f_fut = fut_ts.values()[start_idx - icl : start_idx + ocl] if fut_ts else np.zeros((icl + ocl, 0))
            if len(f_fut) != (icl + ocl):
                continue
            
            x_f = f_fut # (icl + ocl, n_fut)
            x_s = target_ts.static_covariates_values().flatten() # (n_stat)
            
            # Flatten and combine
            flat = np.concatenate([x_p.flatten(), x_f.flatten(), x_s.flatten()])
            inputs_list.append(flat)
        except Exception as e:
            # Likely edge case (start of June without enough history) skip
            continue

    X_june = np.array(inputs_list)
    print(f"Total test windows for June: {X_june.shape[0]}")

    # Device Handling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build Wrapper
    wrapper = TFTWrapper(model, icl, ocl, n_past_feat, n_fut_feat, n_stat_feat)
    wrapper.to(device)
    wrapper.float() # Ensure model is float32
    wrapper.eval()

    # Background Data
    # Use random sampling for background to avoid bias
    bg_size = min(10, len(X_june))
    bg_indices = np.random.choice(len(X_june), bg_size, replace=False)
    background = torch.tensor(X_june[bg_indices], dtype=torch.float32).to(device)
    
    # Explaining all available June windows for better global importance
    test_samples = torch.tensor(X_june, dtype=torch.float32).to(device) 

    print(f"Background shape: {background.shape}")
    print(f"Test samples shape: {test_samples.shape}")
    print(f"Background indices sampled: {bg_indices}")

    print("Initializing GradientExplainer...")
    explainer = shap.GradientExplainer(wrapper, background)
    
    print("Computing SHAP values (this may take a minute)...")
    # explaining test_samples
    shap_results = explainer.shap_values(test_samples)
    
    # SHAP values result is usually a list for multiple outputs. 
    # Since we ensured (batch, 1) output, we take the first element.
    if isinstance(shap_results, list):
        shap_values = np.array(shap_results[0])
    else:
        shap_values = np.array(shap_results)
        
    # Ensure shape is (batch, n_features)
    if len(shap_values.shape) == 3 and shap_values.shape[2] == 1:
        shap_values = shap_values[:, :, 0]
        
    # SHAP values shape: (batch, n_features)
    
    # Feature Names for Plotting
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

    # Exclude target and padded features (if any)
    # The user asked to exclude "padded features and the target column"
    include_indices = [i for i, name in enumerate(feat_names) if target_col not in name]
    
    filtered_shap = shap_values[:, include_indices]
    filtered_names = [feat_names[i] for i in include_indices]
    filtered_vals = test_samples.numpy()[:, include_indices]

    # Diagnostic prints
    print("SHAP shape:", filtered_shap.shape)
    print("Feature count:", len(filtered_names))

    # Global Importance Plot with Aggregation
    print("Aggregating SHAP values by base feature...")
    aggregated_importance = {}
    
    for i, name in enumerate(filtered_names):
        base_name = get_base_feature_name(name)
            
        if base_name not in aggregated_importance:
            aggregated_importance[base_name] = np.zeros(filtered_shap.shape[0])
            
        # Sum SHAP values across time steps for each sample
        aggregated_importance[base_name] += filtered_shap[:, i]

    # Calculate mean absolute importance per base feature
    feature_importance_list = []
    for base_name, sample_shaps in aggregated_importance.items():
        # Mean absolute impact across samples
        mean_abs_importance = np.abs(sample_shaps).mean()
        # Filter out negligible/padded features (all zero importance)
        if mean_abs_importance > 1e-10:
            feature_importance_list.append({"feature": base_name, "importance": mean_abs_importance})

    importance_df = pd.DataFrame(feature_importance_list)
    if importance_df.empty:
        print("Warning: No important features found after filtering.")
    else:
        importance_df = importance_df.sort_values("importance", ascending=False)
        top_features = importance_df.head(20)

        print("Generating Global Aggregated SHAP Importance plot...")
        plt.figure(figsize=(12, 8))
        plt.barh(top_features["feature"], top_features["importance"], color='skyblue', edgecolor='navy')
        plt.gca().invert_yaxis()
        plt.title("Global SHAP Feature Importance (June Aggregate)")
        plt.xlabel("Mean |SHAP value|")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("shap_global_importance_june.png")
        plt.close()

    # Local Impact Plot
    print("Generating Local Explanation plot...")
    plt.figure(figsize=(16, 12))
    
    # Try to get expected_value from explainer, fallback to computing it
    base_val = getattr(explainer, 'expected_value', None)
    if base_val is None:
        with torch.no_grad():
            base_val = wrapper(background).mean().item()
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[0]
        
    # Local Explanation Aggregation
    # Instead of raw lag features, aggregate the first sample's SHAP values
    local_aggregated = {}
    for i, name in enumerate(filtered_names):
        base_name = get_base_feature_name(name)
        if base_name not in local_aggregated:
            local_aggregated[base_name] = 0
        local_aggregated[base_name] += filtered_shap[0, i]
        
    local_names = list(local_aggregated.keys())
    local_vals = np.array(list(local_aggregated.values()))
    
    print(f"Local Aggregated features: {len(local_names)}")
    
    # Use only top features for the plot to avoid clutter
    # Sort for cleaner display if waterfall doesn't do it automatically
    shap.plots._waterfall.waterfall_legacy(base_val, local_vals, feature_names=local_names, max_display=20, show=False)
    plt.title("Top 20 Local Impact Features (Aggregated) for June Prediction")
    plt.savefig("shap_local_impact_june.png", bbox_inches='tight')
    plt.close()

    print("SHAP explainability completed successfully.")
    print("Files generated: shap_global_importance_june.png, shap_local_impact_june.png")

if __name__ == "__main__":
    generate_shap_june()
