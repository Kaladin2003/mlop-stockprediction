import requests
import xml.etree.ElementTree as ET
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# Όρια του σκορ compound για θετικό / αρνητικό
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

# Οικονομικό λεξικό για το VADER
FINANCE_LEXICON = {
    # Θετικοί όροι
    "surge": 2.5, "surges": 2.5, "surged": 2.5, "soar": 2.5, "soars": 2.5, "soared": 2.5,
    "skyrocket": 3.0, "skyrockets": 3.0, "jump": 2.0, "jumps": 2.0, "jumped": 2.0,
    "rally": 2.0, "rallies": 2.0, "rallied": 2.0, "rebound": 1.5, "rebounds": 1.5,
    "rise": 1.5, "rises": 1.5, "rising": 1.5, "rose": 1.5, "climb": 1.5, "climbs": 1.5, "climbed": 1.5,
    "gain": 1.5, "gains": 1.5, "gained": 1.5, "beat": 2.0, "beats": 2.0, "tops": 1.5,
    "exceeds": 2.0, "exceeded": 2.0, "outperform": 2.0, "outperforms": 2.0,
    "upgrade": 2.0, "upgrades": 2.0, "upgraded": 2.0, "bullish": 2.5, "record": 1.5,
    "profit": 1.5, "profits": 1.5, "profitable": 1.5, "growth": 1.5, "expands": 1.2, "expansion": 1.2,
    "buyback": 1.5, "dividend": 1.0, "breakthrough": 2.0, "approval": 1.5, "approved": 1.5,
    # Αρνητικοί όροι
    "fall": -1.5, "falls": -1.5, "fell": -1.5, "falling": -1.5, "drop": -1.5, "drops": -1.5, "dropped": -1.5,
    "decline": -1.5, "declines": -1.5, "declined": -1.5, "slump": -2.0, "slumps": -2.0, "slumped": -2.0,
    "plunge": -2.5, "plunges": -2.5, "plunged": -2.5, "tumble": -2.0, "tumbles": -2.0, "tumbled": -2.0,
    "sink": -1.5, "sinks": -1.5, "sank": -1.5, "slide": -1.5, "slides": -1.5, "selloff": -2.5, "sell-off": -2.5,
    "miss": -2.0, "misses": -2.0, "missed": -2.0, "downgrade": -2.0, "downgrades": -2.0, "downgraded": -2.0,
    "bearish": -2.5, "underperform": -2.0, "underperforms": -2.0, "layoffs": -2.0, "lawsuit": -2.0,
    "probe": -1.5, "investigation": -1.5, "recall": -1.5, "slowdown": -1.5, "bankruptcy": -3.0,
    "default": -2.0, "warns": -1.5, "tariff": -1.0, "tariffs": -1.0, "fined": -2.0,
    # Ουδέτεροι στα οικονομικά (το VADER τους θεωρεί θετικούς)
    "share": 0.0, "shares": 0.0, "stock": 0.0, "fine": 0.0,
}
analyzer.lexicon.update(FINANCE_LEXICON)


def make_analyzer(finance_lexicon=True):
    new_analyzer = SentimentIntensityAnalyzer()
    if finance_lexicon:
        new_analyzer.lexicon.update(FINANCE_LEXICON)
    return new_analyzer


def classify_score(score):
    if score > POSITIVE_THRESHOLD:
        return "positive"
    if score < NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"

NEWS_LIMIT = 5


def fetch_news_titles(ticker, limit=NEWS_LIMIT):
    # Πρώτα yfinance, αλλιώς η ροή RSS του Yahoo Finance
    titles = []
    try:
        for article in (yf.Ticker(ticker).news or [])[:limit]:
            title = article.get('title')
            content = article.get('content')
            if not title and isinstance(content, dict):
                title = content.get('title', '')
            if title:
                titles.append(title)
    except Exception as e:
        print(f"yfinance news error για {ticker}: {e}")
    if titles:
        return titles, "yfinance"

    try:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        titles = [item.findtext("title", "").strip() for item in root.iter("item")]
        titles = [t for t in titles if t][:limit]
    except Exception as e:
        print(f"RSS news error για {ticker}: {e}")
        titles = []
    return titles, ("yahoo_rss" if titles else "none")


def get_news_sentiment(ticker):
    try:
        titles, source = fetch_news_titles(ticker)
        if not titles:
            return {"score": 0.0, "label": "Καμία Είδηση (Ουδέτερο)", "headlines": [], "source": source}
        
        headlines = []
        for title in titles:
            score = analyzer.polarity_scores(title)['compound']
            headlines.append({"title": title, "score": round(score, 2)})
                
        avg_score = sum(h["score"] for h in headlines) / len(headlines)
        
        if avg_score > POSITIVE_THRESHOLD:
            label = "Θετικές Ειδήσεις "
        elif avg_score < NEGATIVE_THRESHOLD:
            label = "Αρνητικές Ειδήσεις "
        else:
            label = "Ουδέτερες Ειδήσεις "
            
        return {"score": round(avg_score, 2), "label": label, "headlines": headlines, "source": source}
    except Exception as e:
        return {"score": 0.0, "label": "Σφάλμα ανάγνωσης ειδήσεων", "headlines": [], "source": "error"}
