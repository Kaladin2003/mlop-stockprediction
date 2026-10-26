import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, precision_score,
                             recall_score, f1_score, roc_auc_score, confusion_matrix)
import json
import time
import os
import mlflow
import mlflow.sklearn

from features import FEATURES
from evaluation_utils import (baseline_metrics, calibration_metrics, save_calibration_plot,
                              save_comparison_plot, save_confusion_matrix_plot,
                              save_feature_importance_plot)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "sp500_cross_sectional")
mlflow.set_tracking_uri(TRACKING_URI)

REPORTS_DIR = os.getenv("REPORTS_DIR", "reports")


N_ESTIMATORS = 200
MAX_DEPTH = 12
MIN_SAMPLES_LEAF = 100

# Οι πρώτες 80% ημέρες για εκπαίδευση
TRAIN_FRACTION = 0.8


def make_random_forest():
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        n_jobs=-1,
        random_state=42,
    )


def temporal_split(df):
    # Χρονικός διαχωρισμός ανά ημερομηνία (όχι τυχαίος)
    unique_dates = np.sort(df['Date'].unique())
    split_date = unique_dates[int(len(unique_dates) * TRAIN_FRACTION)]
    train_df = df[df['Date'] < split_date]
    test_df = df[df['Date'] >= split_date]
    return train_df, test_df, pd.Timestamp(split_date)


def train_cross_sectional_model(combined_csv_path):
    if not os.path.exists(combined_csv_path):
        print(f" Σφάλμα: Δεν βρέθηκε το αρχείο '{combined_csv_path}'.")
        return None

    print("Loading MASSIVE combined dataset...")
    df = pd.read_csv(combined_csv_path, parse_dates=['Date'])

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f" Σφάλμα: Λείπουν οι στήλες {missing} από το σύνολο δεδομένων.")
        return None

    train_df, test_df, split_date = temporal_split(df)
    X_train, y_train = train_df[FEATURES], train_df['Target_Direction']
    X_test, y_test = test_df[FEATURES], test_df['Target_Direction']

    print(f"Εκπαίδευση: {len(X_train):,} γραμμές (πριν από {split_date.date()})")
    print(f"Έλεγχος:    {len(X_test):,} γραμμές (από {split_date.date()} και μετά)")

    os.makedirs(REPORTS_DIR, exist_ok=True)

    mlflow.set_experiment("SP500_Cross_Sectional_Prediction")
    
    with mlflow.start_run() as run:
        # Παράμετροι
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("max_depth", MAX_DEPTH)
        mlflow.log_param("min_samples_leaf", MIN_SAMPLES_LEAF)
        mlflow.log_param("features", FEATURES)
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("split_type", "temporal")
        mlflow.log_param("split_date", str(split_date.date()))
        mlflow.log_param("data_last_date", str(df['Date'].max().date()))
        
        print("Starting Random Forest Training...")
        model = make_random_forest()
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, probabilities)

        # Baseline: πάντα η πιο συχνή κλάση
        up_ratio = y_test.mean()
        baseline_accuracy = max(up_ratio, 1 - up_ratio)
        always_up_f1 = f1_score(y_test, np.ones_like(y_test))
        
        # Μετρικές
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("baseline_accuracy", baseline_accuracy)
        mlflow.log_metric("accuracy_minus_baseline", accuracy - baseline_accuracy)
        mlflow.log_metric("always_up_f1", always_up_f1)
        mlflow.log_metric("test_up_ratio", up_ratio)
        mlflow.log_metric("training_time_sec", training_time)
        mlflow.set_tag("data_source", combined_csv_path)

        # Μοντέλα αναφοράς
        print("Υπολογισμός μοντέλων αναφοράς (πιο συχνή κλάση, ορμή, λογιστική παλινδρόμηση)...")
        comparison = {"random_forest": {"accuracy": accuracy, "roc_auc": roc_auc}}
        comparison.update(baseline_metrics(train_df, test_df))
        for name, values in comparison.items():
            if name == "random_forest":
                continue
            mlflow.log_metric(f"baseline_{name}_accuracy", values["accuracy"])
            mlflow.log_metric(f"baseline_{name}_roc_auc", values["roc_auc"])

        # Βαθμονόμηση
        calibration = calibration_metrics(y_test, probabilities)
        for key, value in calibration.items():
            mlflow.log_metric(f"calibration_{key}", value)

        # Γραφήματα ως artifacts και στον φάκελο reports/
        plots = {
            "confusion_matrix.png": lambda p: save_confusion_matrix_plot(confusion_matrix(y_test, predictions), p),
            "feature_importance.png": lambda p: save_feature_importance_plot(model.feature_importances_, p),
            "calibration_curve.png": lambda p: save_calibration_plot(y_test, probabilities, p),
            "baselines_comparison.png": lambda p: save_comparison_plot(comparison, p),
        }
        for filename, make_plot in plots.items():
            path = os.path.join(REPORTS_DIR, filename)
            make_plot(path)
            mlflow.log_artifact(path, artifact_path="evaluation")
        mlflow.log_dict(
            dict(zip(FEATURES, map(float, model.feature_importances_))),
            "evaluation/feature_importance.json",
        )
        mlflow.log_dict(
            {"comparison": comparison, "calibration": calibration},
            "evaluation/evaluation_summary.json",
        )
        mlflow.log_text(classification_report(y_test, predictions), "evaluation/classification_report.txt")
        
        print(f"\nAccuracy: {accuracy * 100:.2f}% | Baseline: {baseline_accuracy * 100:.2f}% "
              f"| ROC-AUC: {roc_auc:.4f} | F1-Score: {f1:.4f} (πάντα 'Άνοδος': {always_up_f1:.4f})")
        print(f"Χρόνος εκπαίδευσης: {training_time:.1f} δευτ.")
        print("\nΣύγκριση με μοντέλα αναφοράς:")
        for name, values in comparison.items():
            print(f"   {name:22s} accuracy={values['accuracy']:.4f}  roc_auc={values['roc_auc']:.4f}")
        print(f"\nΒαθμονόμηση: Brier={calibration['brier_score']:.5f} "
              f"(αναφορά {calibration['brier_reference']:.5f}) | ECE={calibration['ece']:.4f} "
              f"| εύρος πιθανοτήτων {calibration['prob_min']:.3f}-{calibration['prob_max']:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

        # Καταχώρηση στο Model Registry
        model_info = mlflow.sklearn.log_model(
            sk_model=model, 
            name="random_forest_model", 
            registered_model_name=MODEL_NAME,
            input_example=X_test.head(5),
            # Έμπιστος τύπος για τη μορφή skops του MLflow 3.x
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        
        print(f"\n Success! Model logged and registered to MLflow ({TRACKING_URI}).")
        print(f" Γραφήματα αξιολόγησης: φάκελος '{REPORTS_DIR}/' και MLflow (artifacts/evaluation)")

        return {
            "run_id": run.info.run_id,
            "model_version": getattr(model_info, "registered_model_version", None),
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "baseline_accuracy": baseline_accuracy,
            "training_time_sec": training_time,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

if __name__ == "__main__":
    train_cross_sectional_model("data_processed_sp500/sp500_combined_10y.csv")
