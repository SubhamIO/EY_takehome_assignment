"""
Score the trained model on the held-out 10%.

Run AFTER train.py, from the project root:  python src/evaluate.py

Training and evaluation are deliberately separate scripts: train.py produces the
frozen artifacts, this one only consumes them. That mirrors a real deployment -
you don't retrain to score new data, you load the saved pipeline / encoder and
apply the exact same transforms.
"""

import json
import sys
from pathlib import Path

# make 'src' importable so joblib can re-hydrate the pickled preprocessing function
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

import src.preprocessing  # noqa: F401  (registers module path used by the pickled transformer)
from src.preprocessing import RAW_FEATURES
from src.explain import build_tree_explainer, explain_row, global_importance, shap_table

MODELS_DIR = ROOT / "models"
HOLDOUT_PATH = ROOT / "holdout.csv"
HOLDOUT_LABELS_PATH = ROOT / "holdout_labels.csv"


def main():
    # ---- load the frozen artifacts + the hold-out ------------------------
    model = joblib.load(MODELS_DIR / "model.joblib")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    X_holdout = pd.read_csv(HOLDOUT_PATH)  # Cols A-G, exactly like a client upload
    y_true = pd.read_csv(HOLDOUT_LABELS_PATH)["label"]
    print(f"hold-out rows = {len(X_holdout)}   columns = {list(X_holdout.columns)}")

    # ---- prove the SAVED encoder transforms the raw test data ------------
    # this is the imputers + one-hot encoder lifted straight from training. Loading it
    # on its own and calling transform() (never fit) is the guarantee that the test
    # data sees the exact same encoding the model was trained on.
    X_encoded = preprocessor.transform(X_holdout)
    feature_names = preprocessor.named_steps["features"].get_feature_names_out()
    print(f"encoded feature matrix = {X_encoded.shape}  ({len(feature_names)} features)")

    # ---- predict + score --------------------------------------------------
    preds = model.predict(X_holdout)
    proba = model.predict_proba(X_holdout)  # ROC AUC needs probabilities, not labels

    metrics = {
        "accuracy": float((preds == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "f1_macro": float(f1_score(y_true, preds, average="macro")),
        # proba columns follow model.classes_, so pass that as labels to line them up
        "roc_auc_ovr": float(
            roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=model.classes_)
        ),
    }
    print("\nhold-out performance:")
    for k, v in metrics.items():
        print(f"  {k:18s}: {v:.3f}")

    labels_sorted = sorted(model.classes_)
    print("\nclassification report:")
    print(classification_report(y_true, preds, labels=labels_sorted, zero_division=0))

    cm = confusion_matrix(y_true, preds, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
    print("confusion matrix (rows = true, cols = predicted):")
    print(cm_df)

    # ---- persist the evaluation outputs for the dashboard ----------------
    cm_df.to_csv(MODELS_DIR / "confusion_matrix.csv")
    with open(MODELS_DIR / "holdout_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nsaved confusion -> {MODELS_DIR.name}/confusion_matrix.csv")
    print(f"saved metrics   -> {MODELS_DIR.name}/holdout_metrics.json")

    # ---- capture SHAP for later agentic explanations ---------------------
    # Global importance (mean |SHAP| per original column) feeds the Tab 1 overview.
    # We also run one on-demand explanation to prove the exact path the agent uses.
    explainer = build_tree_explainer(model)
    imp = global_importance(model, preprocessor, X_holdout, explainer=explainer)
    imp.to_csv(MODELS_DIR / "shap_global_importance.csv", index=False)
    print(f"saved shap gimp -> {MODELS_DIR.name}/shap_global_importance.csv")
    print("\ntop columns by mean |SHAP|:")
    print(imp.to_string(index=False))

    demo = explain_row(X_holdout.iloc[[0]], model, preprocessor, explainer)
    print("\nexample per-row explanation (what the agent will receive):")
    print(json.dumps(demo, indent=2))

    # ---- write the full per-row results table ----------------------------
    # layout: all raw features | SHAP per feature | actual label | predicted label.
    # SHAP is for each row's predicted class; Col2 is dropped by the model, so its
    # contribution is 0 by definition (kept in for a uniform, complete table).
    shap_df = shap_table(X_holdout, model, preprocessor, explainer=explainer).round(5)
    results = X_holdout.copy()
    for col in RAW_FEATURES:
        results[f"shap_{col}"] = shap_df.get(f"shap_{col}", 0.0)
    results["actual_label"] = y_true.values
    results["predicted_label"] = preds
    results.to_csv(MODELS_DIR / "holdout_predictions.csv", index=False)
    print(f"\nsaved results   -> {MODELS_DIR.name}/holdout_predictions.csv  ({results.shape[0]} rows)")


if __name__ == "__main__":
    main()
