import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

from features import FEATURES

TARGET = "Target_Direction"


def baseline_metrics(train_df, test_df):
    y_train = train_df[TARGET]
    y_test = test_df[TARGET]
    results = {}

    # 1. Πιο συχνή κλάση του συνόλου εκπαίδευσης
    majority_class = int(y_train.mean() >= 0.5)
    results["majority_class"] = {
        "accuracy": accuracy_score(y_test, np.full(len(y_test), majority_class)),
        "roc_auc": 0.5,
    }

    # 2. Ορμή: αύριο ίδια κατεύθυνση με σήμερα
    momentum_pred = (test_df["Daily_Return"] > 0).astype(int)
    results["momentum"] = {
        "accuracy": accuracy_score(y_test, momentum_pred),
        "roc_auc": roc_auc_score(y_test, test_df["Daily_Return"]),
    }

    # 3. Λογιστική Παλινδρόμηση
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    logreg.fit(train_df[FEATURES], y_train)
    logreg_prob = logreg.predict_proba(test_df[FEATURES])[:, 1]
    results["logistic_regression"] = {
        "accuracy": accuracy_score(y_test, (logreg_prob >= 0.5).astype(int)),
        "roc_auc": roc_auc_score(y_test, logreg_prob),
    }
    return results


def calibration_metrics(y_true, prob, n_bins=10):
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    up_ratio = y_true.mean()

    # ECE με ομάδες ίσου πλήθους (quantile bins)
    edges = np.unique(np.quantile(prob, np.linspace(0, 1, n_bins + 1)))
    bin_ids = np.clip(np.searchsorted(edges, prob, side="right") - 1, 0, len(edges) - 2)
    ece = 0.0
    for b in np.unique(bin_ids):
        mask = bin_ids == b
        ece += mask.mean() * abs(prob[mask].mean() - y_true[mask].mean())

    return {
        "brier_score": brier_score_loss(y_true, prob),
        "brier_reference": up_ratio * (1 - up_ratio),
        "ece": ece,
        "prob_min": float(prob.min()),
        "prob_max": float(prob.max()),
        "prob_std": float(prob.std()),
    }


def save_calibration_plot(y_true, prob, path, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=n_bins, strategy="quantile")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]})
    lo, hi = min(mean_pred.min(), frac_pos.min()) - 0.02, max(mean_pred.max(), frac_pos.max()) + 0.02
    ax1.plot([lo, hi], [lo, hi], "--", color="gray", label="Τέλεια βαθμονόμηση")
    ax1.plot(mean_pred, frac_pos, "o-", label="Random Forest")
    ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi)
    ax1.set_xlabel("Μέση προβλεπόμενη πιθανότητα ανόδου")
    ax1.set_ylabel("Πραγματική συχνότητα ανόδου")
    ax1.set_title("Διάγραμμα Αξιοπιστίας (Calibration)")
    ax1.legend()
    ax2.hist(prob, bins=50)
    ax2.set_xlabel("Προβλεπόμενη πιθανότητα ανόδου")
    ax2.set_ylabel("Πλήθος")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_comparison_plot(results, path):
    names = list(results.keys())
    labels = {
        "random_forest": "Random Forest",
        "majority_class": "Πιο συχνή κλάση",
        "momentum": "Ορμή (χθες)",
        "logistic_regression": "Λογιστική Παλ.",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in [(axes[0], "accuracy", "Accuracy"), (axes[1], "roc_auc", "ROC-AUC")]:
        values = [results[n][metric] for n in names]
        bars = ax.bar([labels.get(n, n) for n in names], values)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(min(values) - 0.03, max(values) + 0.03)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Σύγκριση με μοντέλα αναφοράς (σύνολο ελέγχου)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_confusion_matrix_plot(cm, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    labels = ["Πτώση (0)", "Άνοδος (1)"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Πρόβλεψη"); ax.set_ylabel("Πραγματική τιμή")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
    ax.set_title("Πίνακας Σύγχυσης (σύνολο ελέγχου)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_feature_importance_plot(importances, path):
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(np.array(FEATURES)[order], np.array(importances)[order])
    ax.set_xlabel("Σημαντικότητα (Gini importance)")
    ax.set_title("Σημαντικότητα Χαρακτηριστικών")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
