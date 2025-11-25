# FPL Model Training & Transfer Advisor

AI-powered Fantasy Premier League assistant with position-specific prediction models and transfer recommendations.

## Features

- **User-Friendly Transfer Advisor**: Analyze your FPL team and get AI-powered suggestions
- **Position-Specific Models**: Separate ML models for GK, DEF, MID, FWD
- **Dual Data Sources**: Combines official FPL stats + API-Football detailed match data
- **Smart Recommendations**: Transfer suggestions, optimal lineup, captain picks
- **Admin Dashboard**: Raw data inspection and model analysis

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Data Pipeline
```bash
# Fetch FPL data
python -m fpl_app.ingest_fpl

# Fetch API-Football data
python -m fpl_app.ingest_api_football

# Map players between sources
python -m fpl_app.integrate_players

# Build features
python -m fpl_app.features

# Train models
python -m fpl_app.train_models

# Generate predictions
python -m fpl_app.predict
```

### 3. Launch User App
```bash
streamlit run fpl_app/app.py
```

### 4. Launch Admin Dashboard (Optional)
```bash
streamlit run fpl_app/admin_dashboard.py
```

## Usage

### Transfer Advisor (Main App)
1. Open the app in your browser
2. Enter your FPL ID (find it in your FPL URL)
3. Click "ANALYZE TEAM"
4. Get:
   - Current squad analysis
   - Transfer suggestions with alternatives
   - Optimal starting XI and formation
   - Top 5 captain picks

### Admin Dashboard
- **Predictions**: View all players sorted by predicted points
- **Player Inspector**: Deep dive into individual player stats (FPL + API-Football + Model inputs)
- **Model Inspector**: Examine model weights and feature importance

## Data Sources

- **FPL Official API**: Player stats, prices, fixtures, teams
- **API-Football**: Detailed match statistics (shots, passes, tackles, etc.)

## Configuration

Edit `config.py` to change:
- `SEASON`: Current season (default: 2025 for 2025/26)
- `ROLLING_WINDOWS`: Feature rolling windows (default: [3, 5])
- `API_FOOTBALL_KEY`: Your API key

## Project Structure

```
fpl_app/
├── app.py                  # Main user-friendly transfer advisor
├── admin_dashboard.py      # Admin dashboard for data inspection
├── ingest_fpl.py          # Fetch FPL data
├── ingest_api_football.py # Fetch API-Football data
├── integrate_players.py   # Map players between sources
├── features.py            # Feature engineering
├── train_models.py        # Train position-specific models
├── predict.py             # Generate predictions
├── config.py              # Configuration
├── data/                  # Data storage
└── models/                # Trained models
```

## Models

Each position has a dedicated PyTorch MLP model trained on:
- Rolling averages (3 & 5 games) of FPL stats
- Rolling averages of API-Football detailed stats
- Next fixture context (home/away, opponent strength)
- Player price

Position-specific features:
- **GK**: Saves, clean sheets, goals conceded
- **DEF**: Clean sheets, tackles, interceptions, blocks, goals, assists
- **MID**: Goals, assists, key passes, shots, dribbles, creativity
- **FWD**: Goals, assists, shots, threat, influence

## License

MIT
