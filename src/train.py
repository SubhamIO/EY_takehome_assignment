"""
Train the classifier and save everything the dashboard needs.

Run from the project root:  python src/train.py

Steps, in order:
  1. load the Excel data
  2. clean the messy labels, then fold the ultra-rare classes into 'Other'
  3. carve off a stratified 10% hold-out and save it (Cols A-G only) for the live Tab 2 demo
  4. grid-search the RandomForest with stratified 5-fold CV, scoring on macro-F1 (NOT raw
     accuracy - with ~88% in one class, accuracy is a trap)
  5. save the tuned model AND the fitted preprocessor (imputers + one-hot encoder)

Scoring the hold-out lives in a separate script (evaluate.py) on purpose: training
produces the artifacts, evaluation only consumes them - the same split a real
deployment has.
"""

import json
import sys
from pathlib import Path

# make 'src' importable no matter where we're launched from, and ensure the saved
# pipeline pickles its custom function under a stable module path (src.preprocessing)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import RAW_FEATURES, build_search, clean_labels, group_rare_labels

DATA_PATH = ROOT / "Challenge_Data.xlsx"
MODELS_DIR = ROOT / "models"
TRAIN_PATH = ROOT / "train.csv"
HOLDOUT_PATH = ROOT / "holdout.csv"
HOLDOUT_LABELS_PATH = ROOT / "holdout_labels.csv"
RANDOM_STATE = 42


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    # ---- 1. load -----------------------------------------------------------
    df = pd.read_excel(DATA_PATH)
    print(f"loaded {DATA_PATH.name}: rows, columns = {df.shape}")

    # ---- 2. clean + group the label ---------------------------------------
    df = clean_labels(df)
    df["label"] = group_rare_labels(df["label"])
    print("\nclass distribution used for modelling:")
    print(df["label"].value_counts())
    print((df["label"].value_counts(normalize=True) * 100).round(2).astype(str) + " %")

    X = df[RAW_FEATURES].copy()
    y = df["label"].copy()

    # ---- 3. stratified 90 / 10 split + save the hold-out ------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\ntrain rows = {len(X_train)}   hold-out rows = {len(X_test)}")

    # the brief says the hold-out upload has Cols A-G only, so save exactly that
    X_test.to_csv(HOLDOUT_PATH, index=False)
    # keep the true labels separately, only so WE can score the live demo afterwards
    y_test.to_csv(HOLDOUT_LABELS_PATH, index=False)
    # save the training features too, so the SHAP plots can compare train vs test
    X_train.to_csv(TRAIN_PATH, index=False)
    print(f"saved hold-out features -> {HOLDOUT_PATH.name} (Cols A-G, no label)")
    print(f"saved hold-out labels   -> {HOLDOUT_LABELS_PATH.name} (for our own scoring)")
    print(f"saved train features    -> {TRAIN_PATH.name} (Cols A-G, for SHAP plots)")

    # ---- 4. hyper-parameter tuning (grid search) --------------------------
    # build_search() (in preprocessing.py) wraps the pipeline in a GridSearchCV that
    # tunes the RandomForest through the pipeline, scores folds on macro-F1, and
    # refits the winning config on all the training data.
    search = build_search(random_state=RANDOM_STATE)
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_idx = search.best_index_
    # roc_auc_ovr = one-vs-rest ROC AUC, macro-averaged (right variant for >2 classes)
    metric_names = ["accuracy", "balanced_accuracy", "f1_macro", "roc_auc_ovr"]
    cv_means = {
        m: float(search.cv_results_[f"mean_test_{m}"][best_idx])
        for m in metric_names
    }
    print("\nbest hyper-parameters (chosen by macro-F1):")
    for k, v in search.best_params_.items():
        print(f"  {k} = {v}")
    print("\n5-fold CV of the best model (mean):")
    for m in metric_names:
        print(f"  {m:18s}: {cv_means[m]:.3f}")

    # ---- 5. persist the model + the fitted preprocessor ------------------
    labels_sorted = sorted(y.unique())
    joblib.dump(best_model, MODELS_DIR / "model.joblib")

    # Also save JUST the fitted preprocessing (imputers + one-hot encoder) as its own
    # object, so the evaluation script can load it and transform the hold-out itself.
    # best_model is already fitted, so dropping the final ('model') step leaves a
    # ready-to-transform preprocessor.
    preprocessor = Pipeline(best_model.steps[:-1])
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

    artifacts = {
        "classes": labels_sorted,
        "best_params": search.best_params_,
        "cv_means": cv_means,
        "n_train": int(len(X_train)),
        "n_holdout": int(len(X_test)),
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(artifacts, f, indent=2)

    print(f"\nsaved model        -> {MODELS_DIR.name}/model.joblib")
    print(f"saved preprocessor -> {MODELS_DIR.name}/preprocessor.joblib")
    print(f"saved train metrics-> {MODELS_DIR.name}/metrics.json")
    print("\ntraining done. run  python src/evaluate.py  to score the hold-out.")


if __name__ == "__main__":
    main()
