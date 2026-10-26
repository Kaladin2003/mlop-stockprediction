from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import yfinance as yf
import os
import math
import time
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from features import add_features, FEATURES
from sentiment import get_news_sentiment

class PredictionRequest(BaseModel):
    ticker: str


app = FastAPI(title="S&P 500 Cross-Sectional Stock API", description="Hybrid API using a S&P 500 generalized ML model and live NLP sentiment.")

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "sp500_cross_sectional")
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "production")

# Βάρος των ειδήσεων στο log-odds (0 = μόνο Random Forest)
NLP_STRENGTH = float(os.getenv("NLP_STRENGTH", "1.5"))

mlflow.set_tracking_uri(TRACKING_URI)

model = None
model_version = None


def load_production_model():
    global model, model_version
    try:
        client = MlflowClient()
        version_info = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        model = mlflow.sklearn.load_model(model_uri)
        model_version = version_info.version
        print(f"✅ Loaded model {MODEL_NAME} v{model_version} (@{MODEL_ALIAS}) from MLflow Registry!")
        return True
    except Exception as e:
        print(f" Warning: Could not load from MLflow Registry. Error: {e}")
        model = None
        model_version = None
        return False


load_production_model()


def fetch_latest_data(ticker):
    # 6 μήνες ώστε να υπάρχουν αρκετές ημέρες για τον SMA_50
    data = yf.download(ticker, period="6mo", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    if data.empty:
         raise ValueError(f"Δεν βρέθηκαν δεδομένα για το ticker: {ticker}.")

    data = add_features(data.sort_index())
    data = data.dropna(subset=FEATURES)
    if data.empty:
        raise ValueError(f"Δεν βρέθηκαν δεδομένα για το ticker: {ticker} (ανεπαρκές ιστορικό).")
    return data

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_version": model_version,
    }


@app.post("/reload-model")
def reload_model():
    if not load_production_model():
        raise HTTPException(status_code=503, detail="Αποτυχία φόρτωσης μοντέλου από το MLflow Registry.")
    return {"status": "reloaded", "model_version": model_version}


@app.post("/predict")
def predict_stock(request: PredictionRequest):
    # Νέα προσπάθεια φόρτωσης αν δεν υπήρχε μοντέλο στην εκκίνηση
    if model is None and not load_production_model():
        raise HTTPException(status_code=503, detail="Το ML μοντέλο δεν είναι φορτωμένο. Δεν υπάρχει μοντέλο σε production στο MLflow.")
    
    ticker = request.ticker.strip().upper()
    # Χρόνος ανά στάδιο σε ms
    timings = {}
    t_start = time.perf_counter()
    try:
        # 1. Πρόβλεψη Random Forest
        t0 = time.perf_counter()
        latest_data = fetch_latest_data(ticker)
        latest_features = latest_data[FEATURES].iloc[-1:]
        timings["market_data_ms"] = (time.perf_counter() - t0) * 1000
        
        t0 = time.perf_counter()
        ml_probabilities = model.predict_proba(latest_features)[0]
        timings["ml_inference_ms"] = (time.perf_counter() - t0) * 1000
        ml_prob_down = float(ml_probabilities[0])
        ml_prob_up = float(ml_probabilities[1])
        
        ml_direction = "Up" if ml_prob_up > ml_prob_down else "Down"
        
        # 2. Συναίσθημα ειδήσεων
        t0 = time.perf_counter()
        sentiment = get_news_sentiment(ticker)
        score = sentiment['score']
        timings["news_sentiment_ms"] = (time.perf_counter() - t0) * 1000
        
        # 3. Συνδυασμός με άθροιση log-odds
        p = min(max(ml_prob_up, 1e-6), 1 - 1e-6)
        ml_log_odds = math.log(p / (1 - p))
        nlp_log_odds = NLP_STRENGTH * score
        combined_prob_up = 1 / (1 + math.exp(-(ml_log_odds + nlp_log_odds)))
        combined_prob_down = 1 - combined_prob_up
        
        # Τελική απόφαση
        if combined_prob_up > combined_prob_down:
            final_direction = "ΑΝΟΔΟΣ "
            final_confidence = round(combined_prob_up * 100, 1)
        else:
            final_direction = "ΠΤΩΣΗ "
            final_confidence = round(combined_prob_down * 100, 1)
            
        if final_confidence >= 65.0:
            confidence_level = "ΙΣΧΥΡΗ"
        elif final_confidence >= 55.0:
            confidence_level = "ΜΕΤΡΙΑ"
        else:
            confidence_level = "ΟΡΙΑΚΗ / ΑΒΕΒΑΙΗ"

        timings["total_ms"] = (time.perf_counter() - t_start) * 1000
        timings = {k: round(v, 1) for k, v in timings.items()}
        print(f"/predict {ticker}: {timings}")

        return {
            "ticker": ticker,
            "prediction_ml": "Άνοδος" if ml_direction == "Up" else "Πτώση",
            "ml_confidence_percent": round(max(ml_prob_up, ml_prob_down) * 100, 1),
            "latest_close_price": round(float(latest_data['Close'].iloc[-1]), 2),
            "nlp_sentiment_score": score,
            "nlp_sentiment_label": sentiment['label'],
            "final_direction": final_direction,
            "final_confidence": final_confidence,
            "confidence_level": confidence_level,
            "nlp_strength": NLP_STRENGTH,
            "news_headlines": sentiment.get('headlines', []),
            "news_source": sentiment.get('source', 'none'),
            "nlp_effect_points": round((combined_prob_up - ml_prob_up) * 100, 1),
            "model_version": model_version,
            "timings_ms": timings,
        }
    except Exception as e:
        # Άγνωστο ticker -> 404
        if "Δεν βρέθηκαν " in str(e):
             raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
