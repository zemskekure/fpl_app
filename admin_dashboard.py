import streamlit as st
import pandas as pd
import torch
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from fpl_app import config

st.set_page_config(page_title="FPL AI Dashboard", layout="wide", page_icon="⚽")

# Position and Team Mappings
POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

@st.cache_data
def load_team_names():
    teams = pd.read_csv(config.DATA_DIR / "teams_fpl.csv")
    return dict(zip(teams['id'], teams['name']))

TEAM_NAMES = load_team_names()

# Load Data Caching
@st.cache_data
def load_predictions():
    path = config.DATA_DIR / "predictions_next_gw.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data
def load_features_df():
    path = config.DATA_DIR / "full_features.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data
def load_raw_stats():
    fpl = pd.read_csv(config.DATA_DIR / "fpl_gw_stats.csv")
    api = pd.read_csv(config.DATA_DIR / "player_match_stats_api.csv")
    return fpl, api

def load_model_artifacts(pos):
    model_path = config.MODELS_DIR / f"{pos.lower()}_model.pt"
    feat_path = config.MODELS_DIR / f"{pos.lower()}_features.json"
    if not model_path.exists():
        return None, None
    
    model_state = torch.load(model_path)
    with open(feat_path, 'r') as f:
        features = json.load(f)
    return model_state, features

# --- SIDEBAR ---
st.sidebar.title("FPL AI Manager")
page = st.sidebar.radio("Go to", ["Predictions", "Player Inspector", "Model Inspector"])

# --- PREDICTIONS PAGE ---
if page == "Predictions":
    st.title("🔮 Next Gameweek Predictions")
    
    df = load_predictions()
    if df.empty:
        st.error("No predictions found. Run 'python3 fpl_app/predict.py' first.")
    else:
        # Add readable columns
        df['Position'] = df['element_type'].map(POS_MAP)
        df['Team'] = df['team'].map(TEAM_NAMES)
        df['Price'] = df['value'] / 10  # FPL stores price * 10
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            pos_filter = st.multiselect("Position", df['Position'].unique(), default=df['Position'].unique())
        with col2:
            team_filter = st.multiselect("Team", sorted(df['Team'].unique()), default=df['Team'].unique())
        with col3:
            min_price = st.slider("Min Price (£m)", 4.0, 15.0, 4.0, 0.5)
            
        filtered = df[
            (df['Position'].isin(pos_filter)) & 
            (df['Team'].isin(team_filter)) &
            (df['Price'] >= min_price)
        ].copy()
        
        # Display
        display_df = filtered[['web_name', 'Team', 'Position', 'Price', 'expected_points']].rename(columns={
            'web_name': 'Player',
            'expected_points': 'Predicted Points'
        })
        
        st.dataframe(
            display_df.style.background_gradient(subset=['Predicted Points'], cmap='Greens').format({'Price': '£{:.1f}', 'Predicted Points': '{:.2f}'}),
            use_container_width=True,
            height=800
        )

# --- PLAYER INSPECTOR PAGE ---
elif page == "Player Inspector":
    st.title("🕵️ Player Deep Dive")
    
    preds = load_predictions()
    if preds.empty:
        st.error("No data.")
    else:
        player_name = st.selectbox("Select Player", sorted(preds['web_name'].unique()))
        player_row = preds[preds['web_name'] == player_name].iloc[0]
        pid = player_row['element']
        
        # Header
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Player", player_name)
        with col2:
            st.metric("Team", TEAM_NAMES.get(player_row['team'], 'Unknown'))
        with col3:
            st.metric("Position", POS_MAP.get(player_row['element_type'], 'Unknown'))
        with col4:
            st.metric("Predicted Points", f"{player_row['expected_points']:.2f}")
        
        # Load details
        full_feats = load_features_df()
        fpl_stats, api_stats = load_raw_stats()
        
        # Get mapping to find API ID
        mapping = pd.read_csv(config.DATA_DIR / "player_id_mapping.csv")
        mapped = mapping[mapping['fpl_id'] == pid]
        api_id = mapped.iloc[0]['api_id'] if not mapped.empty else None
        
        tab1, tab2, tab3 = st.tabs(["🎯 Prediction Input", "📊 FPL History", "⚽ API Stats"])
        
        with tab1:
            st.subheader("What the Model Actually Uses for Prediction")
            
            # Get the LAST row for this player (which has next_was_home filled for prediction)
            p_feats = full_feats[full_feats['element'] == pid].sort_values('round').tail(1)
            
            if not p_feats.empty:
                row = p_feats.iloc[0]
                
                # Next Match Context
                st.markdown("### 📅 Next Match Context")
                c1, c2, c3 = st.columns(3)
                with c1:
                    home_away = "Home" if row.get('next_was_home', 0) == 1 else "Away"
                    st.metric("Venue", home_away)
                with c2:
                    opp_str = row.get('next_opponent_strength', 'N/A')
                    st.metric("Opponent Strength", f"{opp_str}/5" if opp_str != 'N/A' else 'N/A')
                with c3:
                    st.metric("Price", f"£{row.get('value', 0)/10:.1f}m")
                
                # Recent Form (Rolling Averages)
                st.markdown("### 📈 Recent Form (Last 3-5 Games)")
                
                form_data = {
                    'Metric': ['Points/Game', 'Minutes/Game', 'Goals/Game', 'Assists/Game', 'xG/Game', 'Shots/Game', 'Key Passes/Game'],
                    'Last 3 GWs': [
                        row.get('total_points_roll_3', 0),
                        row.get('minutes_roll_3', 0),
                        row.get('goals_scored_roll_3', 0),
                        row.get('assists_roll_3', 0),
                        row.get('expected_goals', 0),  # Single game xG
                        row.get('api_shots_total_roll_3', 0),
                        row.get('api_key_passes_roll_3', 0)
                    ],
                    'Last 5 GWs': [
                        row.get('total_points_roll_5', 0),
                        row.get('minutes_roll_5', 0),
                        row.get('goals_scored_roll_5', 0),
                        row.get('assists_roll_5', 0),
                        row.get('expected_goals', 0),
                        row.get('api_shots_total_roll_5', 0),
                        row.get('api_key_passes_roll_5', 0)
                    ]
                }
                
                form_df = pd.DataFrame(form_data)
                st.dataframe(form_df.style.format({'Last 3 GWs': '{:.2f}', 'Last 5 GWs': '{:.2f}'}), use_container_width=True)
                
                # Advanced Metrics
                st.markdown("### 🔬 Advanced Metrics")
                adv_cols = st.columns(4)
                with adv_cols[0]:
                    st.metric("Threat (Roll 5)", f"{row.get('threat_roll_5', 0):.1f}")
                with adv_cols[1]:
                    st.metric("Creativity (Roll 5)", f"{row.get('creativity_roll_5', 0):.1f}")
                with adv_cols[2]:
                    st.metric("ICT Index (Roll 5)", f"{row.get('ict_index_roll_5', 0):.1f}")
                with adv_cols[3]:
                    st.metric("Bonus (Roll 5)", f"{row.get('bonus_roll_5', 0):.1f}")
                
                with st.expander("🔍 View All Raw Features"):
                    st.dataframe(p_feats.T)
            else:
                st.warning("No feature data found.")
                
        with tab2:
            st.subheader("Official FPL Stats (Last 10 GWs)")
            p_fpl = fpl_stats[fpl_stats['element'] == pid].sort_values('round', ascending=False).head(10)
            if not p_fpl.empty:
                display_cols = ['round', 'total_points', 'minutes', 'goals_scored', 'assists', 'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index']
                st.dataframe(p_fpl[display_cols].rename(columns={
                    'round': 'GW',
                    'total_points': 'Pts',
                    'minutes': 'Mins',
                    'goals_scored': 'G',
                    'assists': 'A',
                    'bonus': 'Bonus',
                    'bps': 'BPS',
                    'influence': 'Inf',
                    'creativity': 'Cre',
                    'threat': 'Thr',
                    'ict_index': 'ICT'
                }), use_container_width=True)
            else:
                st.warning("No FPL history found.")
            
        with tab3:
            st.subheader("API-Football Advanced Stats (Last 10 Matches)")
            if api_id:
                p_api = api_stats[api_stats['player_id'] == api_id].head(10)
                if not p_api.empty:
                    display_cols = ['minutes', 'shots_total', 'shots_on', 'goals', 'assists', 'key_passes', 'dribbles_success', 'tackles', 'duels_won']
                    st.dataframe(p_api[display_cols].rename(columns={
                        'shots_total': 'Shots',
                        'shots_on': 'On Target',
                        'goals': 'Goals',
                        'assists': 'Assists',
                        'key_passes': 'Key Passes',
                        'dribbles_success': 'Dribbles',
                        'tackles': 'Tackles',
                        'duels_won': 'Duels Won'
                    }), use_container_width=True)
                else:
                    st.warning("No API stats found.")
            else:
                st.warning("Player not mapped to API-Football.")

# --- MODEL INSPECTOR PAGE ---
elif page == "Model Inspector":
    st.title("🧠 Model Internals")
    
    pos = st.selectbox("Select Position Model", ["GK", "DEF", "MID", "FWD"])
    
    state, features = load_model_artifacts(pos)
    
    if state:
        st.write(f"**Input Features:** {len(features)}")
        
        # Visualise First Layer Weights
        # The first layer is state['net.0.weight'] -> Shape (Hidden_Size, Input_Size)
        # We can take the mean absolute weight for each input feature to see "importance" to the first layer
        
        weights = state['net.0.weight'].numpy() # (64, N_features)
        
        # Mean Absolute Importance
        importance = pd.DataFrame({
            'Feature': features,
            'Mean Weight Magnitude': abs(weights).mean(axis=0)
        }).sort_values('Mean Weight Magnitude', ascending=False)
        
        st.subheader("Feature Importance Proxy (First Layer Weights)")
        st.write("This chart shows which features have the strongest connection weights in the first layer of the neural network.")
        
        fig, ax = plt.subplots(figsize=(10, 12))
        sns.barplot(data=importance.head(30), y='Feature', x='Mean Weight Magnitude', ax=ax, palette='viridis')
        st.pyplot(fig)
        
        with st.expander("View All Feature Weights"):
            st.dataframe(importance)
            
    else:
        st.error(f"Model artifacts for {pos} not found.")
