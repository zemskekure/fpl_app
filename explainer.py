import pandas as pd
import numpy as np
import torch
import joblib
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from fpl_app import config
from fpl_app.train_models import FPLModel

def get_player_feature_importance(player_id, element_type, full_features_df):
    """
    Get scaled feature values (Z-scores) for a player to explain model predictions.
    Returns top contributing features.
    """
    pos_map = {1: 'gk', 2: 'def', 3: 'mid', 4: 'fwd'}
    pos_name = pos_map.get(element_type, 'mid')
    
    # Load model artifacts
    feat_path = config.MODELS_DIR / f"{pos_name}_features.json"
    scaler_path = config.MODELS_DIR / f"{pos_name}_scaler.joblib"
    
    if not feat_path.exists() or not scaler_path.exists():
        return None
        
    with open(feat_path, 'r') as f:
        features = json.load(f)
        
    scaler = joblib.load(scaler_path)
    
    # Get player's latest features
    player_data = full_features_df[full_features_df['element'] == player_id].sort_values('round').tail(1)
    
    if player_data.empty:
        return None
        
    # Extract feature values
    missing_cols = [c for c in features if c not in player_data.columns]
    for c in missing_cols:
        player_data[c] = 0
        
    X = player_data[features].fillna(0).values
    X_scaled = scaler.transform(X)[0]  # Get scaled values (Z-scores)
    
    # Create feature importance dict
    feature_scores = {}
    for i, feat_name in enumerate(features):
        # Z-score shows how many std devs from mean
        # Higher absolute value = more extreme = more influential
        feature_scores[feat_name] = X_scaled[i]
        
    # Sort by absolute value (most extreme features)
    sorted_features = sorted(feature_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    
    return sorted_features[:10]  # Top 10 features

if __name__ == "__main__":
    # Test
    df = pd.read_csv(config.DATA_DIR / "full_features.csv")
    importance = get_player_feature_importance(302, 3, df)  # Example: Doku
    if importance:
        print("Top features for player:")
        for feat, score in importance:
            print(f"{feat}: {score:.2f}")
