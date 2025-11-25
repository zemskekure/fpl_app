import pandas as pd
import unicodedata
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fpl_app import config

def normalize_name(name):
    if not isinstance(name, str):
        return ""
    # Normalize unicode characters to closest ASCII
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')
    return name.lower().strip()

def map_teams(fpl_teams, api_teams):
    # Simple manual mapping or name matching
    
    mapping = {}
    fpl_teams['norm_name'] = fpl_teams['name'].apply(normalize_name)
    api_teams['norm_name'] = api_teams['name'].apply(normalize_name)
    
    # Manual overrides for common mismatches
    overrides = {
        "man utd": "manchester united",
        "man city": "manchester city",
        "spurs": "tottenham",
        "nott'm forest": "nottingham forest",
        "sheffield utd": "sheffield united",
        "luton": "luton town",
        "leicester city": "leicester",
        "ipswich town": "ipswich"
    }
    # Removed "wolves": "wolverhampton" as API has "wolves"

    print("\n--- Team Mapping ---")
    for idx, fpl_row in fpl_teams.iterrows():
        f_name = fpl_row['norm_name']
        original_f_name = f_name
        
        if f_name in overrides:
            f_name = overrides[f_name]
            
        # Try exact match on norm_name
        match = api_teams[api_teams['norm_name'] == f_name]
        
        if match.empty:
            # Try fuzzy check
            match = api_teams[api_teams['norm_name'].str.contains(f_name) | fpl_teams['norm_name'].str.contains(f_name)]
            
        if not match.empty:
            api_id = match.iloc[0]['id']
            mapping[fpl_row['id']] = api_id
            # print(f"Mapped {fpl_row['name']} -> {match.iloc[0]['name']}")
        else:
            print(f"Warning: Could not map FPL team '{fpl_row['name']}' (norm: {f_name})")
            # Suggest potential matches
            # simple containment
            potentials = api_teams[api_teams['norm_name'].apply(lambda x: f_name in x or x in f_name)]
            if not potentials.empty:
                print(f"  Did you mean: {potentials['name'].tolist()}?")
                
    return mapping

def run_integration():
    print("Loading datasets...")
    try:
        fpl_players = pd.read_csv(config.DATA_DIR / "players_fpl.csv")
        fpl_teams = pd.read_csv(config.DATA_DIR / "teams_fpl.csv")
        api_teams = pd.read_csv(config.DATA_DIR / "teams_api.csv")
        api_stats = pd.read_csv(config.DATA_DIR / "player_match_stats_api.csv")
    except FileNotFoundError as e:
        print(f"Error: Missing data files. Run ingestion scripts first. {e}")
        return

    # 1. Map Teams
    team_map = map_teams(fpl_teams, api_teams)
    api_to_fpl_team = {v: k for k, v in team_map.items()}
    
    print(f"Mapped {len(team_map)} / {len(fpl_teams)} teams.")

    # 2. Prepare Player Lists
    fpl_players['full_name'] = fpl_players['first_name'] + " " + fpl_players['second_name']
    fpl_players['norm_name'] = fpl_players['full_name'].apply(normalize_name)
    fpl_players['web_name_norm'] = fpl_players['web_name'].apply(normalize_name)
    
    api_unique = api_stats[['player_id', 'name', 'team_id']].drop_duplicates()
    api_unique['norm_name'] = api_unique['name'].apply(normalize_name)
    api_unique['fpl_team_id'] = api_unique['team_id'].map(api_to_fpl_team)
    
    # 3. Match Players
    player_mapping = []
    matches_found = 0
    
    print("\n--- Player Mapping ---")
    unmapped = []
    
    for idx, fpl_p in fpl_players.iterrows():
        f_id = fpl_p['id']
        team_id = fpl_p['team']
        norm_name = fpl_p['norm_name']
        web_name = fpl_p['web_name_norm']
        
        if team_id not in team_map:
            # Skip if team not mapped (will be reported in team section)
            continue

        # Filter API players by mapped team
        candidates = api_unique[api_unique['fpl_team_id'] == team_id]
        
        if candidates.empty:
            # print(f"No API players found for FPL team ID {team_id}")
            continue
            
        # 1. Exact Full Name
        match = candidates[candidates['norm_name'] == norm_name]
        
        # 2. Web Name (e.g. 'Salah')
        if match.empty:
            # Handle "B.Fernandes" -> "Fernandes"
            clean_web = web_name.split('.')[-1] if '.' in web_name else web_name
            match = candidates[candidates['norm_name'].str.contains(clean_web) | candidates['norm_name'].apply(lambda x: clean_web in x.split())]
            
        # 3. Reverse Containment
        if match.empty:
            match = candidates[candidates['norm_name'].apply(lambda x: x in norm_name)]
            
        # 4. Fuzzy / Partials (First name + Last name parts)
        if match.empty:
            # Check if second name is in api name
            last_name = normalize_name(fpl_p['second_name'])
            match = candidates[candidates['norm_name'].apply(lambda x: last_name in x)]
            
        # 5. Token Overlap (Jaccard) - Aggressive
        if match.empty:
            fpl_tokens = set(norm_name.split())
            def get_overlap(api_name):
                api_tokens = set(api_name.split())
                intersection = fpl_tokens.intersection(api_tokens)
                return len(intersection) / len(fpl_tokens.union(api_tokens))
            
            candidates['overlap'] = candidates['norm_name'].apply(get_overlap)
            best_match = candidates.sort_values('overlap', ascending=False).head(1)
            if not best_match.empty and best_match.iloc[0]['overlap'] >= 0.4: # Threshold
                 match = best_match

        # --- FALLBACK: Global Search (Ignore Team) ---
        if match.empty:
            # Search across ALL API players
            # This handles transfers or data mismatches (e.g. FPL says Arsenal, API says Chelsea)
            
            # 1. Exact Full Name Global
            global_match = api_unique[api_unique['norm_name'] == norm_name]
            
            # 2. Web Name Global (Specific enough?)
            if global_match.empty:
                 clean_web = web_name.split('.')[-1] if '.' in web_name else web_name
                 if len(clean_web) > 3:
                    global_match = api_unique[api_unique['norm_name'].apply(lambda x: clean_web in x.split())]
                 
            # 3. Reverse Containment Global
            if global_match.empty:
                 global_match = api_unique[api_unique['norm_name'].apply(lambda x: x in norm_name)]

            if not global_match.empty:
                # Use the first match, but maybe log a warning if team differs
                match = global_match.iloc[:1]
                found_team_id = match.iloc[0]['team_id']
                # print(f"  Fallback: Found {fpl_p['web_name']} in API Team {found_team_id} (FPL Team {team_id})")
        
        if not match.empty:
            api_id = match.iloc[0]['player_id']
            player_mapping.append({'fpl_id': f_id, 'api_id': api_id})
            matches_found += 1
        else:
            unmapped.append(f"{fpl_p['full_name']} ({fpl_p['team']})")
            
    mapping_df = pd.DataFrame(player_mapping)
    mapping_df.to_csv(config.DATA_DIR / "player_id_mapping.csv", index=False)
    print(f"Mapped {matches_found}/{len(fpl_players)} players.")
    
    if len(unmapped) > 0:
        print(f"Top 10 unmapped players: {unmapped[:10]}")
        with open(config.DATA_DIR / "unmapped_players.txt", "w") as f:
            f.write("\n".join(unmapped))
        print(f"Full list of unmapped players saved to data/unmapped_players.txt")

if __name__ == "__main__":
    run_integration()
