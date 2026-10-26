import argparse
import os
import time
import numpy as np
import pandas as pd
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score

from features import FEATURES
from evaluation_utils import baseline_metrics, calibration_metrics, TARGET
from sp500_training import make_random_forest, N_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES_LEAF, REPORTS_DIR

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(TRACKING_URI)

DATA_PATH = "data_processed_sp500/sp500_combined_10y.csv"
MODEL_LABELS = {
    "random_forest": "Random Forest",
    "majority_class": "Πιο συχνή κλάση",
    "momentum": "Ορμή (χθες)",
    "logistic_regression": "Λογιστική Παλ.",
}


def run_walkforward(years):
    print("Φόρτωση δεδομένων...")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    rows = []

    for year in years:
        # Εκπαίδευση σε όλα τα προηγούμενα έτη (expanding window)
        train_df = df[df["Date"] < f"{year}-01-01"]
        test_df = df[(df["Date"] >= f"{year}-01-01") & (df["Date"] < f"{year + 1}-01-01")]
        if len(train_df) == 0 or len(test_df) == 0:
            print(f"Χρονιά {year}: δεν υπάρχουν αρκετά δεδομένα, παραλείπεται.")
            continue

        print(f"\n=== Χρονιά ελέγχου {year}: εκπαίδευση σε {len(train_df):,} γραμμές, έλεγχος σε {len(test_df):,} ===")
        start = time.time()
        model = make_random_forest()
        model.fit(train_df[FEATURES], train_df[TARGET])
        prob = model.predict_proba(test_df[FEATURES])[:, 1]
        y_test = test_df[TARGET]

        results = {"random_forest": {
            "accuracy": accuracy_score(y_test, (prob >= 0.5).astype(int)),
            "roc_auc": roc_auc_score(y_test, prob),
        }}
        results.update(baseline_metrics(train_df, test_df))
        calibration = calibration_metrics(y_test, prob)

        for name, values in results.items():
            rows.append({
                "year": year,
                "model": name,
                "accuracy": values["accuracy"],
                "roc_auc": values["roc_auc"],
                "test_rows": len(test_df),
                "test_up_ratio": y_test.mean(),
                "brier_score": calibration["brier_score"] if name == "random_forest" else np.nan,
            })
            print(f"   {name:22s} accuracy={values['accuracy']:.4f}  roc_auc={values['roc_auc']:.4f}")
        print(f"   (χρόνος: {time.time() - start:.1f} δευτ.)")

    return pd.DataFrame(rows)


def save_walkforward_plot(table, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, title in [(axes[0], "accuracy", "Accuracy ανά χρονιά"), (axes[1], "roc_auc", "ROC-AUC ανά χρονιά")]:
        for name, group in table.groupby("model", sort=False):
            style = "o-" if name == "random_forest" else "s--"
            ax.plot(group["year"], group[metric], style, label=MODEL_LABELS.get(name, name),
                    linewidth=2.5 if name == "random_forest" else 1.2)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Χρονιά ελέγχου")
        ax.set_xticks(sorted(table["year"].unique()))
    axes[0].legend()
    fig.suptitle("Αξιολόγηση walk-forward (expanding window)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Αξιολόγηση walk-forward του μοντέλου")
    parser.add_argument("--years", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025, 2026])
    args = parser.parse_args()

    table = run_walkforward(args.years)
    if table.empty:
        print("Δεν προέκυψαν αποτελέσματα.")
        raise SystemExit(1)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    csv_path = os.path.join(REPORTS_DIR, "walkforward_results.csv")
    png_path = os.path.join(REPORTS_DIR, "walkforward_results.png")
    table.to_csv(csv_path, index=False)
    save_walkforward_plot(table, png_path)

    # Μέσος όρος και τυπική απόκλιση ανά μοντέλο
    summary = table.groupby("model", sort=False)[["accuracy", "roc_auc"]].agg(["mean", "std"])
    print("\n=== Σύνοψη (μέσος όρος ± τυπική απόκλιση σε όλες τις χρονιές) ===")
    print(summary.round(4).to_string())

    # Καταγραφή στο MLflow, με step = χρονιά
    mlflow.set_experiment("SP500_WalkForward_Evaluation")
    with mlflow.start_run():
        mlflow.log_param("years", args.years)
        mlflow.log_param("window", "expanding")
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("max_depth", MAX_DEPTH)
        mlflow.log_param("min_samples_leaf", MIN_SAMPLES_LEAF)
        for _, row in table.iterrows():
            mlflow.log_metric(f"{row['model']}_accuracy", row["accuracy"], step=int(row["year"]))
            mlflow.log_metric(f"{row['model']}_roc_auc", row["roc_auc"], step=int(row["year"]))
        for name in summary.index:
            mlflow.log_metric(f"{name}_accuracy_mean", summary.loc[name, ("accuracy", "mean")])
            mlflow.log_metric(f"{name}_roc_auc_mean", summary.loc[name, ("roc_auc", "mean")])
        mlflow.log_artifact(csv_path, artifact_path="walkforward")
        mlflow.log_artifact(png_path, artifact_path="walkforward")

    print(f"\nΑποτελέσματα: {csv_path}, {png_path} και MLflow (experiment SP500_WalkForward_Evaluation)")
