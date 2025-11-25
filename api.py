from fastapi import FastAPI, BackgroundTasks
from pathlib import Path
import sys
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fpl_app import ingest_fpl, ingest_api_football, integrate_players, features, train_models, predict, config

app = FastAPI(title="FPL Model API")

def run_full_pipeline():
    ingest_fpl.run_ingestion()
    ingest_api_football.run_ingestion()
    integrate_players.run_integration()
    features.build_features()
    train_models.run_training()
    predict.predict_next_gw()

@app.get("/")
def root():
    return {"message": "FPL Model API Ready"}

@app.post("/ingest")
def ingest_data(background_tasks: BackgroundTasks):
    background_tasks.add_task(ingest_fpl.run_ingestion)
    background_tasks.add_task(ingest_api_football.run_ingestion)
    return {"message": "Ingestion started in background"}

@app.post("/features")
def build_features(background_tasks: BackgroundTasks):
    background_tasks.add_task(features.build_features)
    return {"message": "Feature engineering started"}

@app.post("/train")
def train(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_models.run_training)
    return {"message": "Training started"}

@app.post("/predict")
def trigger_prediction(background_tasks: BackgroundTasks):
    background_tasks.add_task(predict.predict_next_gw)
    return {"message": "Prediction started"}

@app.get("/predictions")
def get_predictions(limit: int = 50):
    path = config.DATA_DIR / "predictions_next_gw.csv"
    if not path.exists():
        return {"error": "No predictions found. Run /predict first."}
    
    df = pd.read_csv(path)
    return df.head(limit).to_dict(orient="records")
