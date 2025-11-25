import streamlit as st
import pandas as pd
import requests
from pathlib import Path
import sys

# --- 1. SETUP & CONFIG ---
sys.path.append(str(Path(__file__).resolve().parent.parent))
from fpl_app import config

st.set_page_config(
    page_title="FPL_TERM",
    page_icon="█",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. STYLES ---
st.markdown("""
<style>
    :root { --bg-color: #F3F4F6; --text-color: #111827; --accent-good: #059669; --accent-bad: #DC2626; --accent-info: #2563EB; --border-color: #000000; }
    .stApp { background-color: var(--bg-color) !important; font-family: 'Courier New', Courier, monospace !important; }
    * { color: var(--text-color) !important; font-family: 'Courier New', Courier, monospace !important; letter-spacing: -0.5px; }
    .good { color: var(--accent-good) !important; font-weight: bold; }
    .bad { color: var(--accent-bad) !important; font-weight: bold; }
    
    /* HIGHLIGHT COLORS */
    .highlight-gold {
        background-color: #FFD700 !important;
        color: #000 !important;
        border-color: #B8860B !important;
    }
    /* Primary Button (Gold - Active/New) */
    .stButton button[kind="primary"] {
        background-color: #FFD700 !important;
        color: #000 !important;
        border: 2px solid #B8860B !important;
    }
    /* Secondary Button (White - Standard) Hover Effect */
    .stButton button[kind="secondary"]:hover {
        border-color: #FFD700 !important;
        color: #B8860B !important;
        background-color: #FFF8DC !important; /* Cornsilk (Light Gold) */
        transform: scale(1.02);
    }
    
    /* PITCH VIZ */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #458B55;
        background-image: linear-gradient(0deg, transparent 24%, rgba(255,255,255,.1) 25%, rgba(255,255,255,.1) 26%, transparent 27%, transparent 74%, rgba(255,255,255,.1) 75%, rgba(255,255,255,.1) 76%, transparent 77%, transparent), linear-gradient(90deg, transparent 24%, rgba(255,255,255,.1) 25%, rgba(255,255,255,.1) 26%, transparent 27%, transparent 74%, rgba(255,255,255,.1) 75%, rgba(255,255,255,.1) 76%, transparent 77%, transparent);
        background-size: 50px 50px;
        border: 5px solid #fff !important;
        border-radius: 10px;
        padding: 20px;
        box-shadow: inset 0 0 50px rgba(0,0,0,0.3);
    }
    .stButton button { background-color: rgba(255,255,255,0.95) !important; border: 2px solid #111 !important; color: #111 !important; border-radius: 5px !important; font-weight: 900 !important; padding: 8px 4px !important; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    div[data-testid="stPopoverBody"] { border: 3px solid #000; background: #fff; }
    div[data-testid="stDataFrame"] { border: 2px solid #000; }
    .box { border: 2px solid var(--border-color); background: #FFFFFF; padding: 15px; margin-bottom: 15px; box-shadow: 4px 4px 0px rgba(0,0,0,0.1); }
    .title-box { border: 3px solid var(--border-color); padding: 20px; background: #FFFFFF; text-align: center; margin-bottom: 30px; box-shadow: 6px 6px 0px rgba(0,0,0,0.2); }
</style>
""", unsafe_allow_html=True)

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        pred = pd.read_csv(config.DATA_DIR / "predictions_next_gw.csv")
        pred = pred.rename(columns={'element': 'id', 'expected_points': 'xp', 'value': 'cost'})
        teams = pd.read_csv(config.DATA_DIR / "teams_fpl.csv")
        team_map = dict(zip(teams['id'], teams['name']))
        pred['team_name'] = pred['team'].map(team_map)
        fixtures = pd.read_csv(config.DATA_DIR / "fpl_fixtures.csv")
        upcoming = fixtures[fixtures['finished'] == False].sort_values('event')
        fix_map = {}
        for tid in teams['id']:
            next_games = upcoming[(upcoming['team_h'] == tid) | (upcoming['team_a'] == tid)]
            if not next_games.empty:
                game = next_games.iloc[0]
                if game['team_h'] == tid:
                    fix_map[tid] = {'opp': team_map.get(game['team_a'], "UNK"), 'diff': game['team_h_difficulty'], 'loc': 'H'}
                else:
                    fix_map[tid] = {'opp': team_map.get(game['team_h'], "UNK"), 'diff': game['team_a_difficulty'], 'loc': 'A'}
            else:
                fix_map[tid] = {'opp': '-', 'diff': 0, 'loc': '-'}
        fpl_stats = pd.read_csv(config.DATA_DIR / "fpl_gw_stats.csv")
        api_stats = pd.read_csv(config.DATA_DIR / "player_match_stats_api.csv")
        boot = requests.get(f"{config.FPL_BASE_URL}/bootstrap-static/").json()
        return pred, boot, fix_map, fpl_stats, api_stats
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return None, None, None, None, None

# Stat tooltips/explanations
STAT_TOOLTIPS = {
    'XP': 'Expected Points: Predicted FPL points for next gameweek based on ML model analyzing form, fixtures, and historical data',
    'Form (Pts/G)': 'Average FPL points per game over the last 5 matches',
    'Mins/G': 'Average minutes played per game over the last 5 matches',
    'Selected By': 'Percentage of all FPL managers who currently own this player',
    'ICT': 'ICT Index: Official FPL metric combining Influence (impact on match), Creativity (chance creation), and Threat (goal threat)',
    'Saves/G': 'Average saves made per game over the last 5 matches',
    'Clean Sheets': 'Total clean sheets (no goals conceded) in the last 5 matches',
    'xGI/G': 'Expected Goal Involvements per game: Statistical measure of attacking contribution (xG + xA) over last 5 matches',
    'GI': 'Goal Involvements: Total goals + assists in the last 5 matches',
    'xG/G': 'Expected Goals per game: Statistical measure of goal-scoring chance quality over last 5 matches',
    'Goals': 'Total goals scored in the last 5 matches',
    'PROJECTED POINTS': 'Total expected points for your optimized Starting XI in the next gameweek',
    'BANK REMAINING': 'Available budget after current squad selection (£0.1m = 1 unit)',
    'TRANSFERS LEFT': 'Number of free transfers remaining this gameweek (additional transfers cost -4 points each)'
}

def get_player_metrics(pid, role, fpl_stats, api_stats):
    p_fpl = fpl_stats[fpl_stats['element'] == pid].sort_values('round', ascending=False).head(5)
    metrics = {}
    metrics['Form (Pts/G)'] = p_fpl['total_points'].mean()
    metrics['Mins/G'] = p_fpl['minutes'].mean()
    # Convert to percentage: selected is in basis points (1% = 100), so divide by 100
    metrics['Selected By'] = f"{p_fpl['selected'].iloc[0] / 100:.1f}%" if not p_fpl.empty else "0%"
    metrics['ICT'] = p_fpl['ict_index'].mean()
    if role == 1: # GK
        metrics['Saves/G'] = p_fpl['saves'].mean()
        metrics['Clean Sheets'] = p_fpl['clean_sheets'].sum()
    elif role == 2: # DEF
        metrics['Clean Sheets'] = p_fpl['clean_sheets'].sum()
        metrics['xGI/G'] = (p_fpl['expected_goals'] + p_fpl['expected_assists']).mean()
    elif role == 3: # MID
        metrics['GI'] = p_fpl['goals_scored'].sum() + p_fpl['assists'].sum()
        metrics['xG/G'] = p_fpl['expected_goals'].mean()
    elif role == 4: # FWD
        metrics['Goals'] = p_fpl['goals_scored'].sum()
        metrics['xG/G'] = p_fpl['expected_goals'].mean()
    return metrics, p_fpl

def get_user_details(uid):
    try:
        r = requests.get(f"{config.FPL_BASE_URL}/entry/{uid}/")
        return r.json() if r.status_code == 200 else None
    except: return None

def get_global_transfer_suggestions(squad, pred_df, bank=0):
    suggestions = []
    available_players = pred_df[~pred_df['id'].isin(squad['id'])]
    targets_by_pos = {
        1: available_players[available_players['element_type'] == 1],
        2: available_players[available_players['element_type'] == 2],
        3: available_players[available_players['element_type'] == 3],
        4: available_players[available_players['element_type'] == 4]
    }
    
    # Get baseline Starting XI score
    baseline_xi, _, baseline_score = optimize_lineup(squad)
    
    for idx, player in squad.iterrows():
        max_budget = player['cost'] + bank
        targets = targets_by_pos[player['element_type']]
        valid_targets = targets[targets['cost'] <= max_budget]
        if valid_targets.empty: continue
        
        # Try each valid target
        for _, target in valid_targets.nlargest(10, 'xp').iterrows():  # Check top 10 to find best actual gain
            # Simulate swap
            simulated_squad = squad[squad['id'] != player['id']].copy()
            simulated_squad = pd.concat([simulated_squad, pd.DataFrame([target])], ignore_index=True)
            
            # Re-optimize with new squad
            new_xi, _, new_score = optimize_lineup(simulated_squad)
            
            # Calculate ACTUAL gain (Starting XI improvement)
            actual_gain = new_score - baseline_score
            
            if actual_gain > 0.5:  # Only suggest if it improves Starting XI
                # Recalibrated Confidence for FPL Margins:
                # 0.5 gain = 60% (Baseline)
                # 1.0 gain = 72%
                # 2.0 gain = 97%
                confidence = min(99, int(60 + (actual_gain - 0.5) * 25))
                
                # Generate reasoning
                if target['xp'] > player['xp']:
                    reason = f"{target['web_name']} has {target['xp']:.1f} xP vs {player['web_name']}'s {player['xp']:.1f} xP and will start in the optimized lineup."
                else:
                    reason = f"This swap improves team balance, allowing a stronger Starting XI formation with +{actual_gain:.1f} total xP gain."
                
                suggestions.append({
                    'out_id': player['id'], 'out_name': player['web_name'], 'out_cost': player['cost'],
                    'in_id': target['id'], 'in_name': target['web_name'], 'in_team': target['team_name'], 'in_cost': target['cost'],
                    'gain': actual_gain, 'confidence': confidence, 'reason': reason
                })
                break  # Found best swap for this player, move to next
                
    suggestions_df = pd.DataFrame(suggestions)
    if not suggestions_df.empty:
        return suggestions_df.sort_values('gain', ascending=False).head(5)
    return suggestions_df

def optimize_lineup(squad_df):
    squad_df = squad_df.sort_values('xp', ascending=False)
    gks = squad_df[squad_df['element_type']==1]
    defs = squad_df[squad_df['element_type']==2]
    mids = squad_df[squad_df['element_type']==3]
    fwds = squad_df[squad_df['element_type']==4]
    
    best_xi = None; best_score = -1
    formations = [(3,4,3), (3,5,2), (4,3,3), (4,4,2), (4,5,1), (5,3,2), (5,4,1)]
    
    for d, m, f in formations:
        if len(defs)>=d and len(mids)>=m and len(fwds)>=f:
            xi = pd.concat([gks.head(1), defs.head(d), mids.head(m), fwds.head(f)])
            if xi['xp'].sum() > best_score:
                best_score = xi['xp'].sum(); best_xi = xi
                
    if best_xi is not None:
        bench = squad_df[~squad_df['id'].isin(best_xi['id'])]
        return best_xi, bench, best_score
    return squad_df, pd.DataFrame(), 0

def set_transfer(out_id, in_row, cost_diff):
    # Only allow transfer if we have transfers remaining
    if st.session_state.transfers_remaining > 0:
        st.session_state.pending_transfer = {'out_id': out_id, 'in_row': in_row, 'cost_diff': cost_diff}
        st.session_state.transfers_remaining -= 1

# --- UI COMPONENTS ---
# DIALOG for Player Details
@st.dialog("PLAYER ANALYSIS")
def show_player_card(player, fix_map, fpl_stats, api_stats, is_new=False):
    fix = fix_map.get(player['team'], {})
    
    if is_new: 
        st.info("★ SIMULATED NEW SIGNING")
    
    # Header
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"## {player['web_name']}")
        st.caption(f"{player['team_name']} | £{player['cost']/10}m")
    with c2:
        st.metric("XP", f"{player['xp']:.1f}")

    st.divider()
    
    # Fixture
    diff = fix.get('diff', 3)
    diff_color = "red" if diff > 3 else "green" if diff < 3 else "grey"
    st.markdown(f"**NEXT OPPONENT**")
    st.markdown(f"### {fix.get('opp')} ({fix.get('loc')})")
    st.caption(f"Difficulty: {diff} ({diff_color})")
    
    st.divider()
    
    # Stats with hover tooltips
    metrics, history = get_player_metrics(player['id'], player['element_type'], fpl_stats, api_stats)
    m_cols = st.columns(2)
    for i, (k, v) in enumerate(metrics.items()):
        tooltip = STAT_TOOLTIPS.get(k, '')
        value_str = f"{v:.2f}" if isinstance(v, float) else str(v)
        with m_cols[i % 2]:
            # Use HTML with title attribute for hover tooltip
            st.markdown(f'<span title="{tooltip}">**{k}**: {value_str}</span>', unsafe_allow_html=True)
        
    st.divider()
    st.caption("RECENT FORM (Last 5 Matches)")
    
    # Position-specific columns for recent form
    role = player['element_type']
    if role == 1:  # GK
        form_cols = ['round', 'total_points', 'minutes', 'saves', 'clean_sheets']
    elif role == 2:  # DEF
        form_cols = ['round', 'total_points', 'minutes', 'clean_sheets', 'expected_goals_conceded']
    elif role == 3:  # MID
        form_cols = ['round', 'total_points', 'minutes', 'goals_scored', 'assists']
    else:  # FWD
        form_cols = ['round', 'total_points', 'minutes', 'goals_scored', 'assists']
    
    # Filter to only existing columns
    available_cols = [col for col in form_cols if col in history.columns]
    st.dataframe(history[available_cols].set_index('round'), use_container_width=True)

def render_pitch_grid(starters, bench, fix_map, fpl_stats, api_stats, prefix, highlight_id=None):
    # Wrap in pitch container
    with st.container(border=True):
        st.caption("STARTING XI (OPTIMIZED)")
        
        # Grid Layout
        rows = [
            starters[starters['element_type']==4], # FWD
            starters[starters['element_type']==3], # MID
            starters[starters['element_type']==2], # DEF
            starters[starters['element_type']==1]  # GK
        ]
        
        for r_idx, players in enumerate(rows):
            count = len(players)
            if count == 0: continue
            
            cols = st.columns(count)
            for i, p in enumerate(players.itertuples()):
                with cols[i]:
                    is_new = (p.id == highlight_id)
                    
                    # Button Label (Clean, no emojis)
                    label = f"{p.web_name}\n{p.xp:.1f}"
                    
                    # Button Styling via Type
                    btn_type = "primary" if is_new else "secondary"
                    
                    if st.button(label, key=f"{prefix}_{r_idx}_{i}", type=btn_type, use_container_width=True):
                        show_player_card(p._asdict(), fix_map, fpl_stats, api_stats, is_new)
            
            if r_idx < 3: 
                st.write("")
                
    # BENCH DISPLAY
    if not bench.empty:
        st.caption("BENCH")
        b_cols = st.columns(len(bench))
        for i, p in enumerate(bench.itertuples()):
            with b_cols[i]:
                is_new = (p.id == highlight_id)
                label = f"{p.web_name}\n{p.xp:.1f}"
                btn_type = "primary" if is_new else "secondary"
                
                if st.button(label, key=f"{prefix}_bench_{i}", type=btn_type, use_container_width=True):
                    show_player_card(p._asdict(), fix_map, fpl_stats, api_stats, is_new)

# --- MAIN ---
st.markdown('<div class="title-box"><h1>FPL_ANALYST // V5.0</h1><a href="http://localhost:8501">[ RAW DATA TERMINAL ]</a></div>', unsafe_allow_html=True)
pred_df, boot, fix_map, fpl_stats, api_stats = load_data()

if 'user_squad' not in st.session_state: st.session_state.user_squad = None
if 'bank' not in st.session_state: st.session_state.bank = 0
if 'pending_transfer' not in st.session_state: st.session_state.pending_transfer = None
if 'transfers_remaining' not in st.session_state: st.session_state.transfers_remaining = 1

tab_team, tab_wildcard = st.tabs(["MY TEAM", "AI WILDCARD"])

with tab_team:
    c_in, c_btn = st.columns([3, 1])
    uid = c_in.text_input("MANAGER ID", placeholder="123456", label_visibility="collapsed")
    if c_btn.button("ANALYZE"):
        st.session_state.pending_transfer = None # Reset on new analyze
        st.session_state.transfers_remaining = 1 # Reset transfers
        user_details = get_user_details(uid)
        curr_gw = next((x['id'] for x in boot['events'] if x['is_current']), 1)
        picks = requests.get(f"{config.FPL_BASE_URL}/entry/{uid}/event/{curr_gw}/picks/").json()
        if 'picks' in picks and user_details:
            p_ids = [p['element'] for p in picks['picks']]
            squad = pred_df[pred_df['id'].isin(p_ids)].copy()
            st.session_state.user_squad = squad
            st.session_state.bank = user_details['last_deadline_bank']
        else: st.error("Invalid ID")

    if st.session_state.user_squad is not None:
        # 1. Apply Simulation
        current_squad = st.session_state.user_squad.copy()
        highlight_id = None
        current_bank = st.session_state.bank
        
        if st.session_state.pending_transfer:
            pt = st.session_state.pending_transfer
            current_squad = current_squad[current_squad['id'] != pt['out_id']]
            new_p = pd.DataFrame([pt['in_row']])
            current_squad = pd.concat([current_squad, new_p], ignore_index=True)
            highlight_id = pt['in_row']['id']
            current_bank += pt['cost_diff']
            # st.warning removed to cleaner look? Or keep as info.
            # User wants "nice color", "same color as selected".
            # I'll trust the highlight logic.

        # 2. Optimize
        starters, bench, proj_score = optimize_lineup(current_squad)
        
        # 3. DASHBOARD METRICS (Colors: Black, Green, Blue)
        # Layout: [ XP | BANK | FT ]
        
        m1, m2, m3 = st.columns(3)
        
        # Helper for styled metric box with hover tooltip
        def metric_box(label, value, color_border, color_text="#000"):
            tooltip = STAT_TOOLTIPS.get(label, '')
            return f"""
            <div style="border: 3px solid {color_border}; background: #FFF; padding: 15px; text-align: center; height: 100%; cursor: help;" title="{tooltip}">
                <div style="color: #666; font-size: 0.8rem; font-weight: bold; text-transform: uppercase;">{label}</div>
                <div style="color: {color_text}; font-size: 2.5rem; font-weight: 900; line-height: 1;">{value}</div>
            </div>
            """
            
        with m1:
            st.markdown(metric_box("PROJECTED POINTS", f"{proj_score:.1f}", "#000000"), unsafe_allow_html=True)
        with m2:
            st.markdown(metric_box("BANK REMAINING", f"£{current_bank/10:.1f}m", "#059669", "#059669"), unsafe_allow_html=True)
        with m3:
            st.markdown(metric_box("TRANSFERS LEFT", str(st.session_state.transfers_remaining), "#2563EB", "#2563EB"), unsafe_allow_html=True)
            
        st.write("") # Spacer
        
        # 4. Render Pitch
        render_pitch_grid(starters, bench, fix_map, fpl_stats, api_stats, "team", highlight_id)
        
        # 5. Transfer Suggestions (Interactive Bars)
        st.markdown("### ALGORITHMIC TRANSFER PROTOCOL")
        
        suggestions = get_global_transfer_suggestions(st.session_state.user_squad, pred_df, st.session_state.bank)
        
        if not suggestions.empty:
            for i, row in suggestions.iterrows():
                # Determine if this row is 'active' (selected)
                is_active = False
                if st.session_state.pending_transfer:
                    pt = st.session_state.pending_transfer
                    if pt['out_id'] == row['out_id'] and pt['in_row']['id'] == row['in_id']:
                        is_active = True
                
                # Prepare Data
                in_player_row = pred_df[pred_df['id'] == row['in_id']].iloc[0]
                out_p = st.session_state.user_squad[st.session_state.user_squad['id'] == row['out_id']].iloc[0]
                cost_diff = out_p['cost'] - row['in_cost']
                
                # Label logic
                label = f"OUT: {row['out_name']}   ➔   IN: {row['in_name']}   (+{row['gain']:.1f} XP)"
                
                # Button Type
                btn_type = "primary" if is_active else "secondary"
                
                # Callback
                if st.button(label, key=f"tx_bar_{i}", type=btn_type, use_container_width=True,
                             on_click=set_transfer,
                             args=(row['out_id'], in_player_row, cost_diff)):
                    pass
                
                # Display confidence and reasoning below the button
                st.caption(f"**{row['confidence']}% Confidence** — {row['reason']}")
                st.write("")  # Spacer
                    
        else:
            st.info("NO SIGNIFICANT UPGRADES FOUND")
                        
        # 6. Global Top 10
        st.markdown("### TOP 10 PROJECTED (GLOBAL)")
        market = pred_df.nlargest(10, 'xp').copy()
        market['cost'] = market['cost'] / 10.0
        market['Owned'] = market['id'].isin(current_squad['id']).map({True: '✅', False: ''})
        st.dataframe(market[['Owned', 'web_name', 'team_name', 'element_type', 'cost', 'xp']], use_container_width=True, hide_index=True)

with tab_wildcard:
    st.info("OPTIMAL AI SQUAD (Budget Ignored)")
    wc_squad = pd.concat([pred_df[pred_df['element_type']==1].nlargest(1, 'xp'), pred_df[pred_df['element_type']==2].nlargest(3, 'xp'), pred_df[pred_df['element_type']==3].nlargest(4, 'xp'), pred_df[pred_df['element_type']==4].nlargest(3, 'xp')])
    st.markdown(f"**Projected: {wc_squad['xp'].sum():.1f}**")
    render_pitch_grid(wc_squad, pd.DataFrame(), fix_map, fpl_stats, api_stats, "wc")
