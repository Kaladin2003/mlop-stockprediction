import pandas as pd
import yfinance as yf
import os
import time
import argparse
from io import StringIO
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests

RAW_FOLDER = "data_raw_sp500"
HISTORY_PERIOD = "10y"
BATCH_SIZE = 100            # μετοχές ανά μαζικό αίτημα
NY_TZ = ZoneInfo("America/New_York")


def get_sp500_tickers(folder=RAW_FOLDER):
    print("Fetching S&P 500 tickers from Wikipedia...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        table = pd.read_html(StringIO(response.text))
        
        tickers = table[0]['Symbol'].tolist()
        # BRK.B -> BRK-B για το yfinance
        tickers = [t.replace('.', '-') for t in tickers]
        return tickers
    except Exception as e:
        # Αν αποτύχει η Wikipedia, χρήση των μετοχών που υπάρχουν ήδη
        existing = sorted(f.replace("_10y.csv", "") for f in os.listdir(folder) if f.endswith("_10y.csv")) if os.path.exists(folder) else []
        print(f"Αποτυχία λήψης λίστας από Wikipedia ({e}). Χρήση των {len(existing)} υπαρχουσών μετοχών.")
        return existing


def ticker_filepath(ticker, folder=RAW_FOLDER):
    return os.path.join(folder, f"{ticker}_10y.csv")


def read_last_date(filepath):
    try:
        dates = pd.read_csv(filepath, usecols=["Date"])["Date"]
        return pd.to_datetime(dates).max() if len(dates) else None
    except Exception:
        return None


def drop_unfinished_session(data):
    # Αφαιρεί τη σημερινή γραμμή αν η αγορά δεν έχει κλείσει
    now_ny = datetime.now(NY_TZ)
    market_closed = now_ny.hour > 16 or (now_ny.hour == 16 and now_ny.minute >= 15)
    if not market_closed:
        data = data[data.index.normalize() < pd.Timestamp(now_ny.date())]
    return data


def download_full_history(ticker, folder=RAW_FOLDER):
    data = yf.download(ticker, period=HISTORY_PERIOD, interval='1d', progress=False)
    if data.empty:
        return 0
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = drop_unfinished_session(data)
    data.to_csv(ticker_filepath(ticker, folder))
    return len(data)


def update_sp500_data(tickers, folder=RAW_FOLDER, full_refresh=False):
    os.makedirs(folder, exist_ok=True)
    report = {"tickers_total": len(tickers), "full_downloads": 0, "incremental_updated": 0,
              "new_rows": 0, "failed": []}

    last_dates = {} if full_refresh else {t: read_last_date(ticker_filepath(t, folder)) for t in tickers}
    full_list = [t for t in tickers if full_refresh or last_dates.get(t) is None]
    incremental_list = [t for t in tickers if t not in full_list]

    # 1. Πλήρες ιστορικό για νέες μετοχές ή πλήρη ανανέωση
    if full_list:
        print(f"Πλήρες κατέβασμα ιστορικού για {len(full_list)} μετοχές. This will take time...")
    for i, ticker in enumerate(full_list):
        print(f"[{i+1}/{len(full_list)}] Downloading: {ticker}")
        try:
            rows = download_full_history(ticker, folder)
            if rows:
                report["full_downloads"] += 1
                report["new_rows"] += rows
            else:
                report["failed"].append(ticker)
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            report["failed"].append(ticker)
        # Καθυστέρηση λόγω rate limiting
        if (i+1) % 10 == 0:
            time.sleep(1)

    # 2. Μόνο οι νέες ημέρες για τις υπάρχουσες μετοχές
    if incremental_list:
        start = min(last_dates[t] for t in incremental_list) + timedelta(days=1)
        if start.date() > datetime.now(NY_TZ).date():
            print("Τα δεδομένα είναι ήδη ενημερωμένα.")
        else:
            print(f"Σταδιακή ενημέρωση {len(incremental_list)} μετοχών από {start.date()}...")
            for b in range(0, len(incremental_list), BATCH_SIZE):
                batch = incremental_list[b:b + BATCH_SIZE]
                try:
                    data = yf.download(batch, start=start.strftime("%Y-%m-%d"), interval="1d",
                                       group_by="ticker", progress=False, threads=True)
                except Exception as e:
                    print(f"Αποτυχία μαζικού αιτήματος ({e})")
                    report["failed"].extend(batch)
                    continue
                for ticker in batch:
                    try:
                        if isinstance(data.columns, pd.MultiIndex):
                            if ticker not in data.columns.get_level_values(0):
                                continue
                            new = data[ticker]
                        else:
                            new = data
                        new = drop_unfinished_session(new.dropna(subset=["Close"]))
                        new = new[new.index > last_dates[ticker]]
                        if new.empty:
                            continue
                        # Ίδια σειρά στηλών με το υπάρχον αρχείο
                        path = ticker_filepath(ticker, folder)
                        columns = [c for c in pd.read_csv(path, nrows=0).columns if c != "Date"]
                        new = new[columns]
                        new.index = new.index.strftime("%Y-%m-%d")
                        new.index.name = "Date"
                        new.to_csv(path, mode="a", header=False)
                        report["incremental_updated"] += 1
                        report["new_rows"] += len(new)
                    except Exception as e:
                        print(f"Error updating {ticker}: {e}")
                        report["failed"].append(ticker)
                time.sleep(1)

    print(f"\nFinished. Πλήρη: {report['full_downloads']} | Ενημερωμένες: {report['incremental_updated']} "
          f"| Νέες γραμμές: {report['new_rows']} | Αποτυχίες: {len(report['failed'])}")
    if report["failed"]:
        print(f"Failed tickers: {report['failed']}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Λήψη/ενημέρωση δεδομένων S&P 500")
    parser.add_argument("--full", action="store_true", help="πλήρης επανάληψη λήψης 10ετούς ιστορικού")
    args = parser.parse_args()

    start_time = time.time()
    tickers = get_sp500_tickers()
    update_sp500_data(tickers, full_refresh=args.full)
    end_time = time.time()
    print(f"Total Ingestion Time: {round((end_time - start_time)/60, 2)} minutes.")
