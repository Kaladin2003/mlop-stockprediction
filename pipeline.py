import argparse
import json
import os
import threading
import time
import traceback
from datetime import datetime

import mlflow
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from sp500_ingestion import get_sp500_tickers, update_sp500_data
from sp500_processing import merge_sp500_data
from data_validation import validate_processed_data, print_report
from sp500_training import train_cross_sectional_model
from promote_model import promote_best_model

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
PROCESSED_PATH = "data_processed_sp500/sp500_combined_10y.csv"

TIMEZONE = os.getenv("PIPELINE_TIMEZONE", "America/New_York")
DAILY_CRON = os.getenv("PIPELINE_CRON", "30 17 * * mon-fri")
FULL_REFRESH_CRON = os.getenv("PIPELINE_FULL_REFRESH_CRON", "0 6 * * sat")
RUN_ON_STARTUP = os.getenv("RUN_ON_STARTUP", "false").lower() == "true"

_pipeline_lock = threading.Lock()


def run_pipeline(full_refresh=False):
    if not _pipeline_lock.acquire(blocking=False):
        print("Μια άλλη εκτέλεση της ροής βρίσκεται σε εξέλιξη· παράλειψη.")
        return "SKIPPED"

    started_at = datetime.now()
    durations = {}
    details = {"full_refresh": full_refresh}
    status = "SUCCESS"

    def stage(name, func, *args, **kwargs):
        print(f"\n{'=' * 20} [{datetime.now():%H:%M:%S}] Στάδιο: {name} {'=' * 20}", flush=True)
        t0 = time.time()
        result = func(*args, **kwargs)
        durations[name] = time.time() - t0
        print(f"{name}: {durations[name]:.1f} δευτ.", flush=True)
        return result

    try:
        # 1. Λήψη/ενημέρωση δεδομένων
        tickers = get_sp500_tickers()
        details["ingestion"] = stage("ingestion", update_sp500_data, tickers, full_refresh=full_refresh)

        # 2. Επεξεργασία και υπολογισμός χαρακτηριστικών
        details["processing"] = stage("processing", merge_sp500_data)

        # 3. Έλεγχος ποιότητας δεδομένων: αν αποτύχει, η ροή σταματά ΠΡΙΝ την εκπαίδευση
        passed, report = stage("validation", validate_processed_data, PROCESSED_PATH)
        print_report(passed, report)
        details["validation"] = report
        if not passed:
            status = "FAILED_VALIDATION"
            return status

        # 4. Εκπαίδευση και καταχώρηση νέου μοντέλου στο MLflow
        training = stage("training", train_cross_sectional_model, PROCESSED_PATH)
        details["training"] = training
        if training is None:
            status = "FAILED_TRAINING"
            return status

        # 5. Σύγκριση champion/challenger
        details["promotion"] = stage("promotion", promote_best_model)
        return status
    except Exception as e:
        status = "ERROR"
        details["error"] = f"{e}\n{traceback.format_exc()}"
        print(f"ΣΦΑΛΜΑ στη ροή: {e}")
        traceback.print_exc()
        return status
    finally:
        total = (datetime.now() - started_at).total_seconds()
        print(f"\n{'=' * 60}\nΑποτέλεσμα ροής: {status} | Συνολική διάρκεια: {total / 60:.1f} λεπτά\n{'=' * 60}", flush=True)
        log_pipeline_run(status, started_at, total, durations, details)
        _pipeline_lock.release()


def log_pipeline_run(status, started_at, total, durations, details):
    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment("SP500_Pipeline_Runs")
        with mlflow.start_run(run_name=f"pipeline_{started_at:%Y%m%d_%H%M}"):
            mlflow.set_tag("status", status)
            mlflow.log_param("full_refresh", details.get("full_refresh"))
            mlflow.log_param("started_at", started_at.isoformat(timespec="seconds"))
            mlflow.log_metric("total_duration_sec", total)
            for name, seconds in durations.items():
                mlflow.log_metric(f"{name}_duration_sec", seconds)
            ingestion = details.get("ingestion") or {}
            for key in ["new_rows", "incremental_updated", "full_downloads"]:
                if key in ingestion:
                    mlflow.log_metric(f"ingestion_{key}", ingestion[key])
            if ingestion.get("failed") is not None:
                mlflow.log_metric("ingestion_failed_tickers", len(ingestion["failed"]))
            processing = details.get("processing") or {}
            if processing:
                mlflow.log_metric("dataset_rows", processing["rows"])
                mlflow.log_param("data_last_date", processing["last_date"])
            promotion = details.get("promotion") or {}
            if promotion:
                mlflow.set_tag("promotion_decision", promotion.get("decision"))
                if promotion.get("version"):
                    mlflow.set_tag("candidate_version", promotion["version"])
            mlflow.log_dict(json.loads(json.dumps(details, default=str)), "pipeline_details.json")
    except Exception as e:
        print(f"Αδυναμία καταγραφής στο MLflow: {e}")


def build_scheduler():
    scheduler = BlockingScheduler(timezone=TIMEZONE)
    common = {"max_instances": 1, "coalesce": True, "misfire_grace_time": 3 * 3600}
    scheduler.add_job(run_pipeline, CronTrigger.from_crontab(DAILY_CRON, timezone=TIMEZONE),
                      id="daily_update", name="Καθημερινή ενημέρωση", kwargs={"full_refresh": False}, **common)
    scheduler.add_job(run_pipeline, CronTrigger.from_crontab(FULL_REFRESH_CRON, timezone=TIMEZONE),
                      id="weekly_full_refresh", name="Εβδομαδιαία πλήρης ανανέωση", kwargs={"full_refresh": True}, **common)
    if RUN_ON_STARTUP:
        scheduler.add_job(run_pipeline, id="startup_run", name="Εκτέλεση κατά την εκκίνηση",
                          next_run_time=datetime.now(scheduler.timezone), kwargs={"full_refresh": False})
    return scheduler


def print_schedule(scheduler):
    print(f"Χρονοπρογραμματιστής ενεργός (ζώνη ώρας {TIMEZONE}):")
    for job in scheduler.get_jobs():
        next_run = job.trigger.get_next_fire_time(None, datetime.now(scheduler.timezone))
        print(f"   - {job.name}: επόμενη εκτέλεση {next_run}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Αυτόματη ροή MLOps")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--once", action="store_true", help="μία εκτέλεση τώρα")
    group.add_argument("--schedule", action="store_true", help="συνεχής λειτουργία με χρονοπρογραμματισμό")
    parser.add_argument("--full", action="store_true", help="πλήρης ανανέωση ιστορικού")
    args = parser.parse_args()

    if args.once:
        result = run_pipeline(full_refresh=args.full)
        raise SystemExit(0 if result == "SUCCESS" else 1)

    scheduler = build_scheduler()
    print_schedule(scheduler)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Ο χρονοπρογραμματιστής σταμάτησε.")
