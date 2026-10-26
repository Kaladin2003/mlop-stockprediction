import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
import requests
import pandas as pd
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
REPORTS_DIR = os.getenv("REPORTS_DIR", "reports")
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM", "KO", "XOM", "PFE", "F"]
STAGES = ["market_data_ms", "ml_inference_ms", "news_sentiment_ms", "total_ms", "client_round_trip_ms"]
STAGE_LABELS = {
    "market_data_ms": "Δεδομένα αγοράς",
    "ml_inference_ms": "Μοντέλο ML",
    "news_sentiment_ms": "Ειδήσεις (NLP)",
    "total_ms": "Σύνολο (API)",
    "client_round_trip_ms": "Σύνολο (πελάτης)",
}


def send_request(ticker):
    start = time.perf_counter()
    try:
        response = requests.post(f"{API_URL}/predict", json={"ticker": ticker}, timeout=120)
        round_trip = (time.perf_counter() - start) * 1000
        row = {"ticker": ticker, "status": response.status_code, "client_round_trip_ms": round_trip}
        if response.status_code == 200:
            row.update(response.json().get("timings_ms", {}))
        return row
    except Exception as e:
        return {"ticker": ticker, "status": f"error: {e}", "client_round_trip_ms": (time.perf_counter() - start) * 1000}


def summarize(df):
    # Μέσος, διάμεσος, p95 και μέγιστο ανά στάδιο
    ok = df[df["status"] == 200]
    rows = []
    for stage in STAGES:
        if stage in ok.columns and ok[stage].notna().any():
            values = ok[stage].dropna()
            rows.append({"stage": stage, "mean": values.mean(), "median": values.median(),
                         "p95": values.quantile(0.95), "max": values.max()})
    return pd.DataFrame(rows).set_index("stage")


def save_plot(df, path):
    ok = df[df["status"] == 200]
    stages = [s for s in STAGES if s in ok.columns]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot([ok[s].dropna() for s in stages], showfliers=True)
    ax.set_xticks(range(1, len(stages) + 1))
    ax.set_xticklabels([STAGE_LABELS[s] for s in stages], rotation=15)
    ax.set_ylabel("Χρόνος (ms)")
    ax.set_title(f"Χρόνος απόκρισης ανά στάδιο ({len(ok)} επιτυχημένα αιτήματα)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Μέτρηση χρόνου απόκρισης του API")
    parser.add_argument("--rounds", type=int, default=5, help="επαναλήψεις για κάθε μετοχή")
    parser.add_argument("--concurrency", type=int, default=1, help="ταυτόχρονα αιτήματα (χρήστες)")
    args = parser.parse_args()

    try:
        health = requests.get(f"{API_URL}/health", timeout=10).json()
    except Exception as e:
        print(f"Το API δεν είναι διαθέσιμο στο {API_URL} ({e}).")
        return
    if not health.get("model_loaded"):
        print("Το API δεν έχει φορτωμένο μοντέλο.")
        return

    requests_list = TICKERS * args.rounds
    print(f"Αποστολή {len(requests_list)} αιτημάτων ({args.concurrency} ταυτόχρονα) στο {API_URL}...")
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(send_request, requests_list))
    wall_time = time.perf_counter() - wall_start

    df = pd.DataFrame(rows)
    success = (df["status"] == 200).sum()
    throughput = success / wall_time
    summary = summarize(df)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    suffix = f"c{args.concurrency}"
    raw_path = os.path.join(REPORTS_DIR, f"latency_raw_{suffix}.csv")
    summary_path = os.path.join(REPORTS_DIR, f"latency_summary_{suffix}.csv")
    plot_path = os.path.join(REPORTS_DIR, f"latency_{suffix}.png")
    df.to_csv(raw_path, index=False)
    summary.to_csv(summary_path)
    save_plot(df, plot_path)

    print(f"\nΕπιτυχημένα: {success}/{len(df)} | Συνολικός χρόνος: {wall_time:.1f} s | Ρυθμός: {throughput:.2f} αιτήματα/s")
    print("\nΧρόνος ανά στάδιο (ms):")
    print(summary.round(1).to_string())

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("API_Latency_Benchmark")
    with mlflow.start_run(run_name=f"concurrency_{args.concurrency}"):
        mlflow.log_param("requests", len(df))
        mlflow.log_param("concurrency", args.concurrency)
        mlflow.log_param("tickers", TICKERS)
        mlflow.log_param("model_version", health.get("model_version"))
        mlflow.log_metric("success_rate", success / len(df))
        mlflow.log_metric("throughput_rps", throughput)
        for stage, row in summary.iterrows():
            for stat in ["mean", "median", "p95", "max"]:
                mlflow.log_metric(f"{stage.replace('_ms', '')}_{stat}_ms", row[stat])
        for path in [raw_path, summary_path, plot_path]:
            mlflow.log_artifact(path, artifact_path="latency")
    print(f"\nΑποτελέσματα: φάκελος '{REPORTS_DIR}/' και MLflow (experiment API_Latency_Benchmark)")


if __name__ == "__main__":
    main()
