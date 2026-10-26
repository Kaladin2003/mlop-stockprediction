import os
import json
from datetime import datetime
import numpy as np
import pandas as pd

from features import FEATURES

DATA_PATH = "data_processed_sp500/sp500_combined_10y.csv"
REPORT_PATH = "data_processed_sp500/validation_report.json"

# Όρια ελέγχων
MIN_TICKERS = int(os.getenv("VALIDATION_MIN_TICKERS", "450"))
MAX_STALENESS_DAYS = int(os.getenv("VALIDATION_MAX_STALENESS_DAYS", "5"))
MAX_ROW_DROP = float(os.getenv("VALIDATION_MAX_ROW_DROP", "0.05"))
UP_RATIO_RANGE = (0.40, 0.60)
MAX_EXTREME_RETURNS = 0.001


def validate_processed_data(path=DATA_PATH, report_path=REPORT_PATH):
    checks = {}

    def check(name, passed, detail):
        checks[name] = {"passed": bool(passed), "detail": detail}

    if not os.path.exists(path):
        check("file_exists", False, f"Δεν βρέθηκε το {path}")
        return False, {"checks": checks}

    df = pd.read_csv(path, parse_dates=["Date"])
    required = ["Date", "Ticker_Name", "Target_Direction"] + FEATURES
    missing = [c for c in required if c not in df.columns]
    check("required_columns", not missing, f"λείπουν: {missing}" if missing else "όλες οι στήλες υπάρχουν")
    if missing:
        return False, {"checks": checks}

    # 1. Ελλιπείς ή άπειρες τιμές
    values = df[FEATURES + ["Target_Direction"]]
    bad = int(values.isna().sum().sum() + np.isinf(values.to_numpy(dtype=float)).sum())
    check("no_missing_values", bad == 0, f"{bad} ελλιπείς/άπειρες τιμές")

    # 2. Πλήθος μετοχών
    n_tickers = df["Ticker_Name"].nunique()
    check("enough_tickers", n_tickers >= MIN_TICKERS, f"{n_tickers} μετοχές (ελάχιστο {MIN_TICKERS})")

    # 3. Φρεσκάδα δεδομένων
    last_date = df["Date"].max()
    staleness = (pd.Timestamp(datetime.now().date()) - last_date).days
    check("data_is_fresh", staleness <= MAX_STALENESS_DAYS,
          f"τελευταία ημερομηνία {last_date.date()} ({staleness} ημέρες πριν, όριο {MAX_STALENESS_DAYS})")

    # 4. Διπλότυπες γραμμές
    duplicates = int(df.duplicated(subset=["Ticker_Name", "Date"]).sum())
    check("no_duplicates", duplicates == 0, f"{duplicates} διπλότυπες γραμμές")

    # 5. Ισορροπία στόχου
    up_ratio = float(df["Target_Direction"].mean())
    check("target_balance", UP_RATIO_RANGE[0] <= up_ratio <= UP_RATIO_RANGE[1],
          f"ποσοστό ανοδικών ημερών {up_ratio:.3f} (εύρος {UP_RATIO_RANGE})")

    # 6. Ακραίες αποδόσεις
    extreme = float((df["Daily_Return"].abs() > 0.5).mean())
    check("no_extreme_returns", extreme <= MAX_EXTREME_RETURNS,
          f"{extreme * 100:.4f}% ημέρες με |απόδοση| > 50%")

    # 7. Σύγκριση πλήθους γραμμών με την προηγούμενη επιτυχημένη εκτέλεση
    previous_rows = None
    if os.path.exists(report_path):
        try:
            with open(report_path, encoding="utf-8") as f:
                previous = json.load(f)
            if previous.get("passed"):
                previous_rows = previous.get("rows")
        except Exception:
            pass
    if previous_rows:
        drop = (previous_rows - len(df)) / previous_rows
        check("no_row_drop", drop <= MAX_ROW_DROP, f"{len(df):,} γραμμές (προηγουμένως {previous_rows:,}, μεταβολή {-drop * 100:+.2f}%)")
    else:
        check("no_row_drop", True, f"{len(df):,} γραμμές (καμία προηγούμενη αναφορά)")

    passed = all(c["passed"] for c in checks.values())
    report = {
        "passed": passed,
        "validated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": len(df),
        "tickers": n_tickers,
        "last_date": str(last_date.date()),
        "up_ratio": up_ratio,
        "checks": checks,
    }
    if passed or not os.path.exists(report_path):
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return passed, report


def print_report(passed, report):
    for name, c in report["checks"].items():
        print(f"   [{'OK' if c['passed'] else 'ΑΠΟΤΥΧΙΑ'}] {name}: {c['detail']}")
    print(f"Αποτέλεσμα ελέγχου δεδομένων: {'ΕΠΙΤΥΧΙΑ' if passed else 'ΑΠΟΤΥΧΙΑ'}")


if __name__ == "__main__":
    ok, rep = validate_processed_data()
    print_report(ok, rep)
