import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score

from features import FEATURES


TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "sp500_cross_sectional")
PRODUCTION_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "production")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Κύρια μετρική σύγκρισης: ROC-AUC.
PRIMARY_METRIC = "roc_auc"
MINIMUM_AUC_THRESHOLD = 0.50

mlflow.set_tracking_uri(TRACKING_URI)


def get_latest_version(client, model_name):
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return None
    return max(versions, key=lambda v: int(v.version))


def get_metric(client, run_id, metric_name, default=0.0):
    try:
        return client.get_run(run_id).data.metrics.get(metric_name, default)
    except Exception:
        return default


def get_production_version(client, model_name, alias):
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except Exception:
        return None


def evaluate_champion_on_candidate_test_set(client, candidate, production):
    # Το μοντέλο παραγωγής αξιολογείται στο ίδιο σύνολο ελέγχου με το υποψήφιο
    try:
        candidate_run = client.get_run(candidate.run_id)
        production_run = client.get_run(production.run_id)
        split_date = candidate_run.data.params.get("split_date")
        data_source = candidate_run.data.tags.get("data_source")
        production_split = production_run.data.params.get("split_date")
        if not split_date or not data_source or not os.path.exists(data_source):
            print("Δεν βρέθηκαν τα δεδομένα ελέγχου του υποψηφίου.")
            return None
        if production_split and pd.Timestamp(production_split) > pd.Timestamp(split_date):
            print("Το μοντέλο παραγωγής εκπαιδεύτηκε σε νεότερα δεδομένα.")
            return None

        df = pd.read_csv(data_source, parse_dates=["Date"])
        test_df = df[df["Date"] >= split_date]
        champion = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{production.version}")
        prob = champion.predict_proba(test_df[FEATURES])[:, 1]
        return roc_auc_score(test_df["Target_Direction"], prob)
    except Exception as e:
        print(f"Αποτυχία επαναξιολόγησης του μοντέλου παραγωγής ({e}); χρήση αποθηκευμένων μετρικών.")
        return None


def notify_api_reload():
    try:
        import requests
        response = requests.post(f"{API_URL}/reload-model", timeout=300)
        print(f"Απάντηση API: {response.status_code} {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Το API δεν ειδοποιήθηκε ({e}). Θα φορτώσει το νέο μοντέλο στην επόμενη εκκίνηση.")
        return False


def promote_best_model(force=False):
    client = MlflowClient()

    print(f"Tracking URI: {TRACKING_URI}")
    print(f"Model: {MODEL_NAME}\n")

    # 1. Το υποψήφιο μοντέλο (challenger)
    candidate = get_latest_version(client, MODEL_NAME)
    if candidate is None:
        print("Δεν βρέθηκαν καταχωρημένες εκδόσεις μοντέλου.")
        return {"decision": "no_candidate"}

    candidate_score = get_metric(client, candidate.run_id, PRIMARY_METRIC)
    candidate_acc = get_metric(client, candidate.run_id, "accuracy")
    candidate_baseline = get_metric(client, candidate.run_id, "baseline_accuracy")

    # 2. Το τρέχον μοντέλο παραγωγής (champion)
    production = get_production_version(client, MODEL_NAME, PRODUCTION_ALIAS)
    comparison_method = "first_deployment"

    if production is None:
        production_score = 0.0
        print("Δεν υπάρχει μοντέλο σε production (πρώτη εκτέλεση).")
    else:
        # Το υποψήφιο είναι ήδη σε production
        if production.version == candidate.version:
            print(f"Η έκδοση {candidate.version} είναι ήδη σε production.")
            return {"decision": "already_production", "version": candidate.version}
        same_test_score = evaluate_champion_on_candidate_test_set(client, candidate, production)
        if same_test_score is not None:
            production_score = same_test_score
            comparison_method = "same_test_set"
            client.set_model_version_tag(MODEL_NAME, candidate.version, "champion_auc_on_same_test_set", f"{same_test_score:.4f}")
        else:
            production_score = get_metric(client, production.run_id, PRIMARY_METRIC)
            comparison_method = "stored_metrics"
    client.set_model_version_tag(MODEL_NAME, candidate.version, "comparison_method", comparison_method)

    print(f"Υποψήφιο μοντέλο   (v{candidate.version}): {PRIMARY_METRIC} = {candidate_score:.4f}")
    print(f"Μοντέλο παραγωγής  {'(v' + str(production.version) + ')' if production else '(κανένα)'}: {PRIMARY_METRIC} = {production_score:.4f}"
          f"  [μέθοδος σύγκρισης: {comparison_method}]")
    print(f"Ελάχιστο αποδεκτό κατώφλι: {PRIMARY_METRIC} > {MINIMUM_AUC_THRESHOLD:.4f}")
    print(f"Ακρίβεια υποψηφίου: {candidate_acc:.4f} | Ακρίβεια baseline : {candidate_baseline:.4f}\n")

    beats_baseline = candidate_acc > candidate_baseline
    client.set_model_version_tag(MODEL_NAME, candidate.version, "beats_baseline_accuracy", str(beats_baseline))

    # 3. Η λογική της προώθησης
    beats_incumbent = candidate_score > production_score
    meets_threshold = candidate_score > MINIMUM_AUC_THRESHOLD

    # Χειροκίνητη έγκριση (python promote_model.py --force)
    if force and not (beats_incumbent and meets_threshold):
        print("ΠΡΟΣΟΧΗ: χειροκίνητη προώθηση --force.")
        client.set_model_version_tag(MODEL_NAME, candidate.version, "manual_override", "true")

    result = {
        "version": candidate.version,
        "candidate_score": candidate_score,
        "production_score": production_score,
        "comparison_method": comparison_method,
    }

    if force or (beats_incumbent and meets_threshold):
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=PRODUCTION_ALIAS,
            version=candidate.version,
        )
        client.set_model_version_tag(MODEL_NAME, candidate.version, "evaluation", "promoted")
        client.set_model_version_tag(MODEL_NAME, candidate.version, f"{PRIMARY_METRIC}_at_promotion", f"{candidate_score:.4f}")

        if production is not None:
            client.set_model_version_tag(MODEL_NAME, production.version, "evaluation", "superseded")

        print(f"PROMOTED: η έκδοση {candidate.version} είναι πλέον '{PRODUCTION_ALIAS}'.")
        print(f"URI για φόρτωση: models:/{MODEL_NAME}@{PRODUCTION_ALIAS}")
        result["decision"] = "promoted"
        result["api_reloaded"] = notify_api_reload()
    else:
        if not meets_threshold:
            reason = f"δεν ξεπερνά το ελάχιστο κατώφλι ({candidate_score:.4f} <= {MINIMUM_AUC_THRESHOLD:.4f})"
        else:
            reason = f"δεν ξεπερνά το μοντέλο παραγωγής ({candidate_score:.4f} <= {production_score:.4f})"

        client.set_model_version_tag(MODEL_NAME, candidate.version, "evaluation", "rejected")
        client.set_model_version_tag(MODEL_NAME, candidate.version, "rejection_reason", reason)

        print(f"REJECTED: η έκδοση {candidate.version} {reason}.")
        print("Το μοντέλο παραγωγής παραμένει αμετάβλητο.")
        result["decision"] = "rejected"
        result["reason"] = reason
    return result


if __name__ == "__main__":
    promote_best_model(force="--force" in sys.argv)
