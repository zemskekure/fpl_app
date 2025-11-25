import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fpl_app import config
from fpl_app.train_models import FPLModel

def get_next_fixtures_map():
    # Load fixtures
    fixtures = pd.read_csv(config.DATA_DIR / "fpl_fixtures.csv")
    # Filter for unfinished
    # Note: 'finished' column might be 'finished' or checked via 'finished_provisional'
    # In bootstrap-static 'events' has 'is_current', 'is_next'.
    # But fixtures table from API usually has 'finished'.
    # Let's assume 'finished' column exists or check 'finished_provisional'.
    
    # Actually, let's check if 'finished' exists. If not, use 'kickoff_time'.
    # For simplicity, let's find the lowest Gameweek (event) that is NOT finished.
    # But we can't easily know "finished" status from just the CSV if we didn't save it.
    # 'fpl_fixtures.csv' comes from `fetch_fixtures`. It has 'finished' column.
    
    upcoming = fixtures[fixtures['finished'] == False]
    if len(upcoming) == 0:
        print("No upcoming fixtures found.")
        return {}
        
    # Sort by event, then kickoff
    upcoming = upcoming.sort_values('event')
    
    # Create map: TeamID -> (OpponentID, WasHome, Strength)
    # We need team strengths.
    teams = pd.read_csv(config.DATA_DIR / "teams_fpl.csv")
    team_strength = teams.set_index('id')['strength'].to_dict()
    
    next_fix_map = {}
    
    # Iterate and pick first next fixture for each team
    teams_list = teams['id'].tolist()
    
    for tid in teams_list:
        # Find fixtures where team is home or away
        team_fixtures = upcoming[(upcoming['team_h'] == tid) | (upcoming['team_a'] == tid)]
        if len(team_fixtures) == 0:
            continue
            
        next_fix = team_fixtures.iloc[0]
        
        if next_fix['team_h'] == tid:
            is_home = True
            opp_id = next_fix['team_a']
        else:
            is_home = False
            opp_id = next_fix['team_h']
            
        opp_strength = team_strength.get(opp_id, 3) # Default 3
        
        next_fix_map[tid] = {
            'next_was_home': is_home, # In feature terms 'was_home' (aka 'is_home' for the match to predict)
            'next_opponent_strength': opp_strength,
            'next_event': next_fix['event']
        }
        
    return next_fix_map

def predict_next_gw():
    print("Generating predictions...")
    
    # Load Full Features (History)
    df = pd.read_csv(config.DATA_DIR / "full_features.csv")
    
    # Get Last Row per Player
    df = df.sort_values(['element', 'round'])
    latest_df = df.groupby('element').tail(1).copy()
    
    # Get Next Fixture Info
    next_fix_map = get_next_fixtures_map()
    
    # Update Context columns
    # Map team -> next info
    # latest_df has 'team' column (from merge in build_features)
    
    def fill_next(row):
        tid = row['team']
        if tid in next_fix_map:
            info = next_fix_map[tid]
            row['next_was_home'] = info['next_was_home']
            row['next_opponent_strength'] = info['next_opponent_strength']
            row['next_event'] = info['next_event']
        else:
            # No game?
            row['next_was_home'] = np.nan # Will filter out
        return row
        
    latest_df = latest_df.apply(fill_next, axis=1)
    
    # Filter players with no game
    pred_df = latest_df.dropna(subset=['next_was_home'])
    
    # Merge Names back in (since features only kept IDs)
    players_df = pd.read_csv(config.DATA_DIR / "players_fpl.csv")
    # We need first_name, second_name, web_name
    pred_df = pred_df.merge(players_df[['id', 'first_name', 'second_name', 'web_name']], left_on='element', right_on='id', how='left')
    
    results = []
    
    pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    
    for pid, pname in pos_map.items():
        pos_rows = pred_df[pred_df['element_type'] == pid].copy()
        if len(pos_rows) == 0:
            continue
            
        # Load Model Artifacts
        model_path = config.MODELS_DIR / f"{pname.lower()}_model.pt"
        feat_path = config.MODELS_DIR / f"{pname.lower()}_features.json"
        scaler_path = config.MODELS_DIR / f"{pname.lower()}_scaler.joblib"
        
        if not model_path.exists():
            print(f"Model for {pname} not found.")
            continue
            
        with open(feat_path, 'r') as f:
            features = json.load(f)
            
        scaler = joblib.load(scaler_path)
        
        # Prepare X
        # Ensure columns exist (fill missing rolling with 0 if any, though shouldn't happen if history exists)
        # Check for missing columns in df that are expected by model
        missing_cols = [c for c in features if c not in pos_rows.columns]
        if missing_cols:
            print(f"Warning: Missing columns for {pname}: {missing_cols}")
            for c in missing_cols:
                pos_rows[c] = 0
                
        X = pos_rows[features].fillna(0).values
        X_scaled = scaler.transform(X)
        
        # Load Model
        model = FPLModel(len(features))
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Predict
        with torch.no_grad():
            preds = model(torch.FloatTensor(X_scaled)).numpy().flatten()
            
        pos_rows['expected_points'] = preds
        results.append(pos_rows[['element', 'web_name', 'first_name', 'second_name', 'team', 'element_type', 'next_event', 'expected_points', 'value']])
        
    if len(results) == 0:
        print("No predictions made.")
        return
        
    final_df = pd.concat(results)
    final_df = final_df.sort_values('expected_points', ascending=False)
    
    out_file = config.DATA_DIR / "predictions_next_gw.csv"
    final_df.to_csv(out_file, index=False)
    print(f"Saved predictions for {len(final_df)} players to {out_file.name}")
    print(final_df[['web_name', 'expected_points', 'element_type']].head(20))

if __name__ == "__main__":
    predict_next_gw()
