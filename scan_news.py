from sentiment import get_news_sentiment

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "INTC",
           "ORCL", "CRM", "ADBE", "NFLX", "DIS", "JPM", "BAC", "GS", "V", "MA",
           "WMT", "COST", "KO", "PEP", "MCD", "NKE", "SBUX", "BA", "CAT", "GE",
           "XOM", "CVX", "PFE", "MRK", "LLY", "JNJ", "UNH", "F", "GM", "UBER"]


def main():
    results = []
    sources = {}
    for i, t in enumerate(TICKERS):
        print(f"[{i+1}/{len(TICKERS)}] {t}...   ", end="\r")
        s = get_news_sentiment(t)
        sources[s["source"]] = sources.get(s["source"], 0) + 1
        results.append((s["score"], t, s))

    print(f"\n\nΠηγές ειδήσεων: {sources}")

    results.sort(key=lambda r: r[0], reverse=True)

    def show(rows):
        for score, t, s in rows:
            print(f"\n{t}: σκορ {score:+.2f}  ({s['source']}, {len(s['headlines'])} τίτλοι)")
            for h in s["headlines"]:
                print(f"   {h['score']:+.2f}  {h['title']}")

    print("\n=== ΠΙΟ ΘΕΤΙΚΕΣ ΕΙΔΗΣΕΙΣ ===")
    show(results[:5])
    print("\n=== ΠΙΟ ΑΡΝΗΤΙΚΕΣ ΕΙΔΗΣΕΙΣ ===")
    show(results[-3:])


if __name__ == "__main__":
    main()
