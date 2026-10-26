import os
from datetime import datetime
import pandas as pd

from sentiment import fetch_news_titles
from scan_news import TICKERS

LABEL_FILE = os.path.join("data_sentiment", "headlines_to_label.csv")
HEADLINES_PER_TICKER = 20


def load_label_file(path):
    # Διαχωριστικό ';' (Excel) ή ','
    if not os.path.exists(path):
        return pd.DataFrame(columns=["id", "collected_at", "ticker", "title", "label"])
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if "title" not in df.columns:
        df = pd.read_csv(path, sep=",", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    return df


def main():
    os.makedirs("data_sentiment", exist_ok=True)
    existing = load_label_file(LABEL_FILE)
    seen = set(existing["title"].str.strip().str.lower()) if len(existing) else set()

    new_rows = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    for i, ticker in enumerate(TICKERS):
        print(f"[{i+1}/{len(TICKERS)}] {ticker}...   ", end="\r")
        titles, source = fetch_news_titles(ticker, limit=HEADLINES_PER_TICKER)
        for title in titles:
            key = title.strip().lower()
            if key and key not in seen:
                seen.add(key)
                new_rows.append({"collected_at": now, "ticker": ticker, "title": title.strip(), "label": ""})

    # Τυχαία σειρά για την επισημείωση
    new_df = pd.DataFrame(new_rows).sample(frac=1, random_state=42) if new_rows else pd.DataFrame()
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined["id"] = range(1, len(combined) + 1)
    combined = combined[["id", "collected_at", "ticker", "title", "label"]]
    combined.to_csv(LABEL_FILE, sep=";", index=False, encoding="utf-8-sig")

    labeled = (combined["label"].str.strip() != "").sum()
    print(f"\n\nΝέοι τίτλοι: {len(new_rows)} | Σύνολο: {len(combined)} | Με ετικέτα: {labeled}")
    print(f"Αρχείο: {LABEL_FILE}")


if __name__ == "__main__":
    main()
