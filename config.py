import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# FPL Settings
FPL_BASE_URL = "https://fantasy.premierleague.com/api"

# API-FOOTBALL Settings
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "1996481afb744d896c966edceb11ca51")
API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
SEASON = 2025  # 2025/2026 season
LEAGUE_ID = 39 # Premier League

# Mapping Settings
POSITIONS = {
    1: "GK",
    2: "DEF",
    3: "MID",
    4: "FWD"
}

# Feature Generation Settings
ROLLING_WINDOWS = [3, 5]
