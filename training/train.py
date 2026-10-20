"""Train and evaluate the spam classifier.

Usage:
    python -m training.train
    python -m training.train --data my_emails.csv --text-col body --label-col label

Pipeline
    1. Load + deduplicate (duplicates across train/test inflate scores).
    2. Hold out a stratified test set that is touched exactly once.
    3. Select the model family with 5-fold cross-validation on the train split only.
    4. Pick the decision threshold on out-of-fold predictions so that precision
       meets a target (a false positive means a real email is hidden, which is
       worse than letting one spam through).
    5. Evaluate once on the held-out test set and write models/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "sms_spam_collection.tsv"
MODEL_PATH = ROOT / "models" / "spam_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

RANDOM_STATE = 42
TARGET_PRECISION = 0.99


def load_dataset(path: Path, text_col: str | None, label_col: str | None) -> pd.DataFrame:
    """Load a dataset into columns ``text`` and ``label`` (1 = spam)."""
    if path.suffix == ".tsv" and text_col is None:
        df = pd.read_csv(path, sep="\t", names=["label", "text"], quoting=3, encoding="utf-8")
    else:
        df = pd.read_csv(path)
        df = df.rename(columns={text_col or "text": "text", label_col or "label": "label"})
    df = df[["label", "text"]].dropna()
    if not pd.api.types.is_numeric_dtype(df["label"]):
        df["label"] = df["label"].astype(str).str.strip().str.lower().map(
            {"ham": 0, "spam": 1, "0": 0, "1": 1}
        )
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"Loaded {before} rows, {before - len(df)} duplicates dropped, {len(df)} kept")
    print(f"Spam rate: {df['label'].mean():.3%}")
    return df


def build_features() -> ColumnTransformer:
    """Word n-grams plus character n-grams (catches obfuscation like 'fr33' and URLs)."""
    word = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode"
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, lowercase=True
    )
    return ColumnTransformer([("word", word, "text"), ("char", char, "text")])


def candidates() -> dict[str, Pipeline]:
    return {
        "naive_bayes": Pipeline(
            [
                ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                ("clf", MultinomialNB(alpha=0.1)),
            ]
        ),
        "logreg": Pipeline(
            [
                ("features", build_features()),
                ("clf", LogisticRegression(C=10, max_iter=2000, class_weight="balanced")),
            ]
        ),
        "linear_svm_calibrated": Pipeline(
            [
                ("features", build_features()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        LinearSVC(C=0.5, class_weight="balanced"), cv=3, method="sigmoid"
                    ),
                ),
            ]
        ),
    }


def as_input(name: str, series: pd.Series):
    """Naive Bayes pipeline takes a Series; the others take a DataFrame with a text column."""
    return series if name == "naive_bayes" else series.to_frame("text")


def pick_threshold(y_true: np.ndarray, scores: np.ndarray, target_precision: float) -> float:
    """Lowest threshold whose out-of-fold precision meets the target (maximises recall)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    ok = np.where(precision[:-1] >= target_precision)[0]
    if len(ok) == 0:
        return 0.5
    # Among thresholds that meet the precision target, choose the one with best recall.
    best = ok[np.argmax(recall[:-1][ok])]
    return float(thresholds[best])


def report(y_true, scores, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y_true, pred)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4),
        "pr_auc": round(float(average_precision_score(y_true, scores)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--text-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--target-precision", type=float, default=TARGET_PRECISION)
    args = parser.parse_args()

    df = load_dataset(args.data, args.text_col, args.label_col)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE
    )
    print(f"Train {len(X_train)} / Test {len(X_test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results: dict[str, dict] = {}
    oof_scores: dict[str, np.ndarray] = {}
    for name, pipe in candidates().items():
        scores = cross_val_predict(
            pipe, as_input(name, X_train), y_train, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        oof_scores[name] = scores
        cv_results[name] = {
            "pr_auc": round(float(average_precision_score(y_train, scores)), 4),
            "roc_auc": round(float(roc_auc_score(y_train, scores)), 4),
        }
        print(f"CV  {name:24s} PR-AUC={cv_results[name]['pr_auc']:.4f} ROC-AUC={cv_results[name]['roc_auc']:.4f}")

    best = max(cv_results, key=lambda n: cv_results[n]["pr_auc"])
    threshold = pick_threshold(y_train.to_numpy(), oof_scores[best], args.target_precision)
    cv_report = report(y_train.to_numpy(), oof_scores[best], threshold)
    print(f"\nSelected: {best}  threshold={threshold:.4f} (target precision {args.target_precision})")

    model = candidates()[best]
    model.fit(as_input(best, X_train), y_train)
    test_scores = model.predict_proba(as_input(best, X_test))[:, 1]
    test_report = report(y_test.to_numpy(), test_scores, threshold)
    print("Held-out test:", json.dumps(test_report, indent=2))

    # Refit on all data for the shipped artifact; reported numbers come from the held-out split.
    final = candidates()[best]
    final.fit(as_input(best, df["text"]), df["label"])

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {
            "pipeline": final,
            "model_name": best,
            "threshold": threshold,
            "text_input": "series" if best == "naive_bayes" else "frame",
            "sklearn_version": sklearn.__version__,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
        MODEL_PATH,
        compress=3,
    )

    metrics = {
        "dataset": str(args.data.name),
        "n_rows_after_dedup": int(len(df)),
        "spam_rate": round(float(df["label"].mean()), 4),
        "split": {"train": int(len(X_train)), "test": int(len(X_test)), "seed": RANDOM_STATE},
        "cross_validation_5fold_train_only": cv_results,
        "selected_model": best,
        "target_precision": args.target_precision,
        "cv_at_selected_threshold": cv_report,
        "held_out_test": test_report,
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "caveat": (
            "Trained on the UCI SMS Spam Collection (short English text messages). "
            "Performance on long-form email is not measured; retrain on email data with --data."
        ),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nSaved {MODEL_PATH.relative_to(ROOT)} and {METRICS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
