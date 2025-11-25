import requests
import pandas as pd
import time
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fpl_app import config

HEADERS = {
    'x-rapidapi-host': "v3.football.api-sports.io",
    'x-rapidapi-key': config.API_FOOTBALL_KEY
}

def get_teams():
    print("Fetching teams from API-Football...")
    url = f"{config.API_FOOTBALL_BASE_URL}/teams"
    params = {"league": config.LEAGUE_ID, "season": config.SEASON}
    response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()
    data = response.json()['response']
    
    teams = []
    for item in data:
        team = item['team']
        venue = item['venue']
        team['venue_name'] = venue['name']
        teams.append(team)
    
    return pd.DataFrame(teams)

def get_fixtures():
    print("Fetching fixtures from API-Football...")
    url = f"{config.API_FOOTBALL_BASE_URL}/fixtures"
    params = {"league": config.LEAGUE_ID, "season": config.SEASON}
    response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()
    data = response.json()['response']
    
    fixtures = []
    for item in data:
        f = item['fixture']
        l = item['league']
        t_home = item['teams']['home']
        t_away = item['teams']['away']
        g = item['goals']
        
        record = {
            'fixture_id': f['id'],
            'date': f['date'],
            'timestamp': f['timestamp'],
            'gameweek': int(l['round'].replace("Regular Season - ", "")) if "Regular Season" in l['round'] else None,
            'status': f['status']['short'],
            'home_team_id': t_home['id'],
            'home_team_name': t_home['name'],
            'away_team_id': t_away['id'],
            'away_team_name': t_away['name'],
            'home_goals': g['home'],
            'away_goals': g['away']
        }
        fixtures.append(record)
        
    return pd.DataFrame(fixtures)

def get_fixture_stats(fixture_id):
    url = f"{config.API_FOOTBALL_BASE_URL}/fixtures/players"
    params = {"fixture": fixture_id}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 429:
        print("Rate limit reached. Sleeping for 60 seconds...")
        time.sleep(60)
        response = requests.get(url, headers=HEADERS, params=params)
    
    response.raise_for_status()
    return response.json()['response']

def run_ingestion():
    # 1. Teams
    teams_df = get_teams()
    teams_df.to_csv(config.DATA_DIR / "teams_api.csv", index=False)
    print(f"Saved {len(teams_df)} teams to teams_api.csv")
    
    # 2. Fixtures
    fixtures_df = get_fixtures()
    fixtures_df.to_csv(config.DATA_DIR / "matches_api.csv", index=False)
    print(f"Saved {len(fixtures_df)} fixtures to matches_api.csv")
    
    # 3. Match Stats (Iterate over completed matches)
    # Filter for finished matches to get stats
    finished_matches = fixtures_df[fixtures_df['status'].isin(['FT', 'AET', 'PEN'])]
    fixture_ids = finished_matches['fixture_id'].tolist()
    
    print(f"Fetching stats for {len(fixture_ids)} finished matches...")
    
    all_stats = []
    # Check if we already have some data to avoid re-fetching everything if run multiple times
    output_file = config.DATA_DIR / "player_match_stats_api.csv"
    existing_ids = set()
    if output_file.exists():
        try:
            existing_df = pd.read_csv(output_file)
            if 'fixture_id' in existing_df.columns:
                existing_ids = set(existing_df['fixture_id'].unique())
                all_stats = existing_df.to_dict('records')
                print(f"Loaded {len(existing_ids)} existing matches.")
        except:
            pass

    cnt = 0
    for fid in fixture_ids:
        if fid in existing_ids:
            continue
            
        try:
            data = get_fixture_stats(fid)
            # Data structure: list of 2 teams
            for team_data in data:
                team_id = team_data['team']['id']
                players = team_data['players']
                for p in players:
                    player_info = p['player']
                    stats = p['statistics'][0] # Usually array of 1
                    
                    row = {
                        'fixture_id': fid,
                        'team_id': team_id,
                        'player_id': player_info['id'],
                        'name': player_info['name'],
                        'minutes': stats['games']['minutes'],
                        'rating': stats['games']['rating'],
                        'shots_total': stats['shots']['total'],
                        'shots_on': stats['shots']['on'],
                        'goals': stats['goals']['total'],
                        'assists': stats['goals']['assists'],
                        'saves': stats['goals']['saves'],
                        'passes_total': stats['passes']['total'],
                        'key_passes': stats['passes']['key'],
                        'tackles': stats['tackles']['total'],
                        'blocks': stats['tackles']['blocks'],
                        'interceptions': stats['tackles']['interceptions'],
                        'duels_total': stats['duels']['total'],
                        'duels_won': stats['duels']['won'],
                        'dribbles_attempts': stats['dribbles']['attempts'],
                        'dribbles_success': stats['dribbles']['success'],
                        'fouls_drawn': stats['fouls']['drawn'],
                        'fouls_committed': stats['fouls']['committed']
                    }
                    all_stats.append(row)
            
            cnt += 1
            if cnt % 10 == 0:
                print(f"Fetched stats for {cnt} new matches...")
                # Save intermediate
                pd.DataFrame(all_stats).to_csv(output_file, index=False)
            
            time.sleep(0.2) # Be nice to API
            
        except Exception as e:
            print(f"Error fetching fixture {fid}: {e}")
            
    final_df = pd.DataFrame(all_stats)
    final_df.to_csv(output_file, index=False)
    print(f"Saved stats for {len(final_df)} player-matches to player_match_stats_api.csv")

if __name__ == "__main__":
    run_ingestion()
