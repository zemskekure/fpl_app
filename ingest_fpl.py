import requests
import pandas as pd
import time
from pathlib import Path
import sys

# Add the current directory to sys.path to allow imports if running directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fpl_app import config

def fetch_bootstrap_static():
    """Fetch global FPL data."""
    print("Fetching bootstrap-static...")
    url = f"{config.FPL_BASE_URL}/bootstrap-static/"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def fetch_fixtures():
    """Fetch all fixtures."""
    print("Fetching fixtures...")
    url = f"{config.FPL_BASE_URL}/fixtures/"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def fetch_player_summary(player_id):
    """Fetch gameweek history for a specific player."""
    url = f"{config.FPL_BASE_URL}/element-summary/{player_id}/"
    response = requests.get(url)
    # Simple retry logic
    if response.status_code == 429:
        time.sleep(2)
        response = requests.get(url)
    response.raise_for_status()
    return response.json()

def run_ingestion():
    # 1. Bootstrap Static
    bootstrap = fetch_bootstrap_static()
    
    # Process Players (Elements)
    players_df = pd.DataFrame(bootstrap['elements'])
    players_df.to_csv(config.DATA_DIR / "players_fpl.csv", index=False)
    print(f"Saved {len(players_df)} players to players_fpl.csv")
    
    # Process Teams
    teams_df = pd.DataFrame(bootstrap['teams'])
    teams_df.to_csv(config.DATA_DIR / "teams_fpl.csv", index=False)
    print(f"Saved {len(teams_df)} teams to teams_fpl.csv")

    # Process Events (Gameweeks)
    events_df = pd.DataFrame(bootstrap['events'])
    events_df.to_csv(config.DATA_DIR / "events_fpl.csv", index=False)

    # 2. Fixtures
    fixtures = fetch_fixtures()
    fixtures_df = pd.DataFrame(fixtures)
    fixtures_df.to_csv(config.DATA_DIR / "fpl_fixtures.csv", index=False)
    print(f"Saved {len(fixtures_df)} fixtures to fpl_fixtures.csv")

    # 3. Player Summaries (History)
    print("Fetching player summaries (this may take a while)...")
    all_history = []
    player_ids = players_df['id'].tolist()
    
    for i, pid in enumerate(player_ids):
        try:
            summary = fetch_player_summary(pid)
            history = summary['history']
            # Add player_id to each record
            for record in history:
                record['element'] = pid
            all_history.extend(history)
            
            if i % 50 == 0:
                print(f"Processed {i}/{len(player_ids)} players...")
            
        except Exception as e:
            print(f"Error fetching summary for player {pid}: {e}")
    
    history_df = pd.DataFrame(all_history)
    history_df.to_csv(config.DATA_DIR / "fpl_gw_stats.csv", index=False)
    print(f"Saved {len(history_df)} historical records to fpl_gw_stats.csv")

if __name__ == "__main__":
    run_ingestion()
