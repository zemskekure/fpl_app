import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fpl_app import config

def load_data():
    print("Loading data for feature engineering...")
    data = {}
    try:
        data['fpl_history'] = pd.read_csv(config.DATA_DIR / "fpl_gw_stats.csv")
        data['fpl_players'] = pd.read_csv(config.DATA_DIR / "players_fpl.csv")
        data['fpl_teams'] = pd.read_csv(config.DATA_DIR / "teams_fpl.csv")
        data['api_stats'] = pd.read_csv(config.DATA_DIR / "player_match_stats_api.csv")
        data['api_matches'] = pd.read_csv(config.DATA_DIR / "matches_api.csv")
        data['mapping'] = pd.read_csv(config.DATA_DIR / "player_id_mapping.csv")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    return data

def process_api_stats(api_stats, api_matches):
    stats = api_stats.merge(api_matches[['fixture_id', 'gameweek']], on='fixture_id', how='left')
    agg_dict = {
        'minutes': 'sum', 'shots_total': 'sum', 'shots_on': 'sum', 'goals': 'sum', 'assists': 'sum',
        'saves': 'sum', 'passes_total': 'sum', 'key_passes': 'sum', 'tackles': 'sum', 'blocks': 'sum',
        'interceptions': 'sum', 'duels_total': 'sum', 'duels_won': 'sum', 'dribbles_attempts': 'sum',
        'dribbles_success': 'sum', 'fouls_drawn': 'sum', 'fouls_committed': 'sum'
    }
    valid_aggs = {k: v for k, v in agg_dict.items() if k in stats.columns}
    grouped = stats.groupby(['player_id', 'gameweek']).agg(valid_aggs).reset_index()
    grouped.columns = ['api_id', 'round'] + [f"api_{c}" for c in grouped.columns if c not in ['player_id', 'gameweek']]
    return grouped

def add_rolling_features(df, features, windows=[3, 5]):
    df = df.sort_values(['element', 'round'])
    for feat in features:
        if feat not in df.columns: continue
        for w in windows:
            col_name = f"{feat}_roll_{w}"
            df[col_name] = df.groupby('element')[feat].transform(lambda x: x.rolling(window=w, min_periods=1).mean())
    return df

def build_features():
    data = load_data()
    df = data['fpl_history'].copy()
    players = data['fpl_players'][['id', 'element_type', 'team']]
    df = df.merge(players, left_on='element', right_on='id', how='left')
    
    print("Processing API stats...")
    api_grouped = process_api_stats(data['api_stats'], data['api_matches'])
    mapping = data['mapping']
    df = df.merge(mapping, left_on='element', right_on='fpl_id', how='left')
    df = df.merge(api_grouped, left_on=['api_id', 'round'], right_on=['api_id', 'round'], how='left')
    
    api_cols = [c for c in api_grouped.columns if c.startswith('api_')]
    df[api_cols] = df[api_cols].fillna(0)
    
    teams = data['fpl_teams'][['id', 'strength']]
    df = df.merge(teams, left_on='opponent_team', right_on='id', suffixes=('', '_opp'), how='left')
    df.rename(columns={'strength': 'opponent_strength'}, inplace=True)
    
    fpl_feats_to_roll = ['total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded', 'saves', 'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index', 'value']
    all_roll_feats = fpl_feats_to_roll + api_cols
    
    print("Calculating rolling features...")
    df = add_rolling_features(df, all_roll_feats, config.ROLLING_WINDOWS)
    
    df = df.sort_values(['element', 'round'])
    df['target_points'] = df.groupby('element')['total_points'].shift(-1)
    df['next_was_home'] = df.groupby('element')['was_home'].shift(-1)
    df['next_opponent_strength'] = df.groupby('element')['opponent_strength'].shift(-1)
    
    full_df_path = config.DATA_DIR / "full_features.csv"
    df.to_csv(full_df_path, index=False)
    print(f"Saved full feature set to {full_df_path}")
    
    train_df = df.dropna(subset=['target_points', 'next_was_home', 'next_opponent_strength'])
    context_features = ['element', 'round', 'next_was_home', 'next_opponent_strength', 'value']
    roll_cols = [c for c in df.columns if '_roll_' in c]
    
    def get_pos_features(pos_name, keywords):
        feats = context_features.copy()
        for c in roll_cols:
            if any(k in c for k in keywords): feats.append(c)
        return list(set(feats))
    
    gk_keywords = ['saves', 'clean_sheets', 'conceded', 'minutes', 'points', 'bonus', 'bps', 'ict']
    def_keywords = ['clean_sheets', 'conceded', 'goals', 'assists', 'minutes', 'points', 'bonus', 'tackles', 'interceptions', 'blocks', 'duels', 'bps', 'ict', 'threat', 'creativity']
    mid_keywords = ['goals', 'assists', 'minutes', 'points', 'bonus', 'key_passes', 'shots', 'dribbles', 'passes', 'bps', 'ict', 'threat', 'creativity', 'influence']
    fwd_keywords = ['goals', 'assists', 'minutes', 'points', 'bonus', 'shots', 'key_passes', 'touches', 'bps', 'ict', 'threat', 'influence']
    
    pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    for pid, pname in pos_map.items():
        pos_df = train_df[train_df['element_type'] == pid].copy()
        if pid == 1: feats = get_pos_features('GK', gk_keywords)
        elif pid == 2: feats = get_pos_features('DEF', def_keywords)
        elif pid == 3: feats = get_pos_features('MID', mid_keywords)
        else: feats = get_pos_features('FWD', fwd_keywords)
            
        final_cols = feats + ['target_points']
        final_cols = [c for c in final_cols if c in pos_df.columns]
        out_df = pos_df[final_cols]
        out_df.to_csv(config.DATA_DIR / f"{pname.lower()}_df.csv", index=False)
        print(f"Saved {pname} training data: {len(out_df)} rows")
        with open(config.MODELS_DIR / f"{pname.lower()}_features.json", 'w') as f:
            json.dump([c for c in final_cols if c != 'target_points'], f)

if __name__ == "__main__":
    build_features()
