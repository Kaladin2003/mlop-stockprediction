import io
import os
import zipfile
import requests
import pandas as pd
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from sentiment import make_analyzer, classify_score
from collect_headlines import LABEL_FILE, load_label_file

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(TRACKING_URI)

REPORTS_DIR = os.getenv("REPORTS_DIR", "reports")
PHRASEBANK_DIR = "data_sentiment"
PHRASEBANK_ZIP = os.path.join(PHRASEBANK_DIR, "FinancialPhraseBank-v1.0.zip")
PHRASEBANK_URL = "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"
CLASSES = ["negative", "neutral", "positive"]

# Αποδεκτές ετικέτες στη χειροκίνητη επισημείωση
LABEL_MAP = {
    "positive": "positive", "pos": "positive", "p": "positive", "+": "positive", "1": "positive",
    "θετικό": "positive", "θετικο": "positive", "θ": "positive",
    "negative": "negative", "neg": "negative", "n": "negative", "-": "negative", "-1": "negative",
    "αρνητικό": "negative", "αρνητικο": "negative", "α": "negative",
    "neutral": "neutral", "neu": "neutral", "0": "neutral", "o": "neutral",
    "ουδέτερο": "neutral", "ουδετερο": "neutral", "ο": "neutral",
}


def load_phrasebank():
    # Financial PhraseBank (AllAgree)
    os.makedirs(PHRASEBANK_DIR, exist_ok=True)
    if not os.path.exists(PHRASEBANK_ZIP):
        print("Λήψη Financial PhraseBank από το Hugging Face...")
        try:
            response = requests.get(PHRASEBANK_URL, timeout=60)
            response.raise_for_status()
            with open(PHRASEBANK_ZIP, "wb") as f:
                f.write(response.content)
        except Exception as e:
            print(f"Αποτυχία λήψης ({e}).")
            print(f"Δεν βρέθηκε το αρχείο {PHRASEBANK_ZIP}.")
            return None

    with zipfile.ZipFile(PHRASEBANK_ZIP) as z:
        name = next((n for n in z.namelist() if "AllAgree" in n and n.endswith(".txt")), None)
        if name is None:
            print("Δεν βρέθηκε το αρχείο Sentences_AllAgree.txt μέσα στο zip.")
            return None
        text = z.read(name).decode("latin-1")

    rows = []
    for line in text.splitlines():
        if "@" in line:
            sentence, label = line.rsplit("@", 1)
            label = label.strip().lower()
            if label in CLASSES:
                rows.append({"text": sentence.strip(), "label": label})
    return pd.DataFrame(rows)


def load_manual_labels():
    df = load_label_file(LABEL_FILE)
    if df.empty:
        return None
    df["label"] = df["label"].str.strip().str.lower().map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    if df.empty:
        return None
    return df.rename(columns={"title": "text"})[["text", "label"]]


def evaluate(df):
    # Σύγκριση VADER με και χωρίς οικονομικό λεξικό
    versions = {"vader_original": make_analyzer(finance_lexicon=False),
                "vader_finance": make_analyzer(finance_lexicon=True)}
    results = {}
    for name, analyzer in versions.items():
        predicted = [classify_score(analyzer.polarity_scores(t)["compound"]) for t in df["text"]]
        # Ακρίβεια πολικότητας: μόνο στις μη ουδέτερες προτάσεις
        polar = df["label"] != "neutral"
        results[name] = {
            "predicted": predicted,
            "accuracy": accuracy_score(df["label"], predicted),
            "macro_f1": f1_score(df["label"], predicted, average="macro", labels=CLASSES, zero_division=0),
            "polarity_accuracy": accuracy_score(df.loc[polar, "label"], [p for p, m in zip(predicted, polar) if m]) if polar.any() else float("nan"),
            "confusion": confusion_matrix(df["label"], predicted, labels=CLASSES),
            "report": classification_report(df["label"], predicted, labels=CLASSES, zero_division=0),
        }
    return results


def save_confusion_plots(results, dataset_name, path):
    titles = {"vader_original": "VADER (αρχικό)", "vader_finance": "VADER + οικονομικό λεξικό"}
    labels_gr = ["Αρνητικό", "Ουδέτερο", "Θετικό"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (name, res) in zip(axes, results.items()):
        cm = res["confusion"]
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3)); ax.set_xticklabels(labels_gr)
        ax.set_yticks(range(3)); ax.set_yticklabels(labels_gr)
        ax.set_xlabel("Πρόβλεψη"); ax.set_ylabel("Πραγματική ετικέτα")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(f"{titles[name]}\naccuracy={res['accuracy']:.3f}, macro-F1={res['macro_f1']:.3f}")
    fig.suptitle(f"Αξιολόγηση συναισθήματος: {dataset_name}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    datasets = {}
    phrasebank = load_phrasebank()
    if phrasebank is not None:
        datasets["financial_phrasebank"] = phrasebank
    manual = load_manual_labels()
    if manual is not None:
        datasets["manual_headlines"] = manual
    else:
        print(f"Δεν βρέθηκαν τίτλοι με ετικέτα στο {LABEL_FILE}.")

    if not datasets:
        print("Κανένα σύνολο δεδομένων για αξιολόγηση.")
        return

    mlflow.set_experiment("Sentiment_Evaluation")
    summary_rows = []
    for dataset_name, df in datasets.items():
        print(f"\n=== {dataset_name}: {len(df)} προτάσεις | κατανομή: {df['label'].value_counts().to_dict()} ===")
        results = evaluate(df)
        plot_path = os.path.join(REPORTS_DIR, f"sentiment_{dataset_name}.png")
        save_confusion_plots(results, dataset_name, plot_path)

        with mlflow.start_run(run_name=dataset_name):
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("n_samples", len(df))
            mlflow.log_param("thresholds", "positive > 0.05, negative < -0.05")
            for name, res in results.items():
                for metric in ["accuracy", "macro_f1", "polarity_accuracy"]:
                    mlflow.log_metric(f"{name}_{metric}", res[metric])
                    summary_rows.append({"dataset": dataset_name, "version": name, "metric": metric, "value": res[metric]})
                mlflow.log_text(res["report"], f"sentiment/{name}_report.txt")
                print(f"\n{name}: accuracy={res['accuracy']:.4f} | macro-F1={res['macro_f1']:.4f} "
                      f"| polarity accuracy={res['polarity_accuracy']:.4f}")
                print(res["report"])
            mlflow.log_artifact(plot_path, artifact_path="sentiment")

            # Προτάσεις όπου οι δύο εκδοχές διαφωνούν
            diff = df.assign(vader_original=results["vader_original"]["predicted"],
                             vader_finance=results["vader_finance"]["predicted"])
            diff = diff[diff["vader_original"] != diff["vader_finance"]]
            diff_path = os.path.join(REPORTS_DIR, f"sentiment_{dataset_name}_differences.csv")
            diff.to_csv(diff_path, index=False, encoding="utf-8-sig")
            mlflow.log_artifact(diff_path, artifact_path="sentiment")

    summary = pd.DataFrame(summary_rows).pivot_table(index=["dataset", "metric"], columns="version", values="value")
    summary_path = os.path.join(REPORTS_DIR, "sentiment_summary.csv")
    summary.to_csv(summary_path, encoding="utf-8-sig")
    print("\n=== Σύνοψη ===")
    print(summary.round(4).to_string())
    print(f"\nΑποτελέσματα: φάκελος '{REPORTS_DIR}/' και MLflow (experiment Sentiment_Evaluation)")


if __name__ == "__main__":
    main()
