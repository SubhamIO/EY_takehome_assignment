"""
Save the model-explanation figures used in the report / Tab 1 of the dashboard.

Run after train.py, from the project root:  python src/plots.py

Produces (under models/plots/):
  - rf_feature_importance.png                 random-forest built-in importance, per column
  - shap_global_importance.png                mean |SHAP| per column (prediction-level importance)
  - shap_summary_numeric_<class>_<split>.png  SHAP beeswarm, numeric features, train & test
  - shap_scatter_numeric_<class>_<split>.png  SHAP value vs feature value, train & test

The SHAP views need a single class to explain; we use the meaningful minority
(Category_2) because that's where feature effects are most interesting - Category_1 is
so dominant its curves are almost flat. We plot train and test side by side so we can
eyeball whether the feature effects are stable across the split (a sanity check
against overfitting).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")  # write files only, no interactive window

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import src.preprocessing  # noqa: F401  (registers module path for the pickled transformer)
from src.explain import build_tree_explainer, global_importance, rf_importance_by_column
from src.preprocessing import DATE_COLS, NUMERIC_COLS

MODELS_DIR = ROOT / "models"
PLOTS_DIR = MODELS_DIR / "plots"
HOLDOUT_PATH = ROOT / "holdout.csv"
TRAIN_PATH = ROOT / "train.csv"
TARGET_CLASS = "Category_2"


def _barh(df, value_col, title, path):
    """Small horizontal bar chart, biggest value on top."""
    d = df.sort_values(value_col)  # ascending -> largest ends up at the top in barh
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(d["feature"], d[value_col], color="steelblue")
    ax.set_title(title)
    ax.set_xlabel(value_col)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path.relative_to(ROOT)}")


def _shap_numeric_plots(X, tag, preprocessor, explainer, class_idx, feature_names, numeric_features, num_idx):
    """SHAP beeswarm + dependence scatter for the numeric features, for one dataset."""
    shap_vals = explainer(np.asarray(preprocessor.transform(X))).values  # (rows, features[, classes])
    shap_vals_cls = shap_vals[:, :, class_idx] if shap_vals.ndim == 3 else shap_vals
    # plot against the UNSCALED values so the x-axis / colour reads in real units
    # (Col3 in its raw amount, months 1-12) instead of the RobustScaler output
    X_display = src.preprocessing._prepare_frame(X)[numeric_features]

    shap.summary_plot(shap_vals_cls[:, num_idx], X_display, show=False)
    fig = plt.gcf()
    fig.suptitle(f"SHAP summary - numeric features ({TARGET_CLASS}, {tag})", y=1.02)
    p = PLOTS_DIR / f"shap_summary_numeric_{TARGET_CLASS}_{tag}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p.relative_to(ROOT)}")

    # the direction question: does SHAP rise or fall as the value grows? a clear
    # upward/downward trend means the feature has a monotonic effect on the class.
    n = len(num_idx)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, i, name in zip(axes, num_idx, numeric_features):
        xv = X_display[name].values
        ax.scatter(xv, shap_vals_cls[:, i], s=12, alpha=0.5, color="steelblue")
        ax.axhline(0, color="grey", lw=0.8)  # above = pushes toward the class, below = away
        ax.set_xlabel(f"{name} value")
        ax.set_ylabel("SHAP value")
        ax.set_title(name)
        lo, hi = np.nanpercentile(xv, [1, 99])  # clip Col3's outliers so the trend is visible
        if hi > lo:
            ax.set_xlim(lo, hi)
    for ax in axes[n:]:  # hide any unused subplot
        ax.set_visible(False)
    fig.suptitle(f"SHAP vs feature value - numeric features ({TARGET_CLASS}, {tag})")
    fig.tight_layout()
    p = PLOTS_DIR / f"shap_scatter_numeric_{TARGET_CLASS}_{tag}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p.relative_to(ROOT)}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODELS_DIR / "model.joblib")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    X = pd.read_csv(HOLDOUT_PATH)
    rf = model.named_steps["model"]

    # ---- 1. random-forest built-in importance ----------------------------
    rf_imp = rf_importance_by_column(model, preprocessor)
    rf_imp.to_csv(MODELS_DIR / "rf_feature_importance.csv", index=False)
    _barh(rf_imp, "rf_importance", "RandomForest feature importance (per column)",
          PLOTS_DIR / "rf_feature_importance.png")

    # ---- 2. SHAP global importance ---------------------------------------
    explainer = build_tree_explainer(model)
    shap_imp = global_importance(model, preprocessor, X, explainer=explainer)
    _barh(shap_imp, "mean_abs_shap", "SHAP global importance (mean |SHAP| per column)",
          PLOTS_DIR / "shap_global_importance.png")

    # ---- 3. SHAP for the NUMERIC features only, on BOTH train and test ----
    # beeswarm colour = feature value, which only means something for genuinely numeric
    # features. The ordinal-encoded text columns would colour by an arbitrary code, so
    # we restrict these SHAP views to Col3 + the engineered date parts.
    feature_names = [n.split("__", 1)[-1] for n in preprocessor.named_steps["features"].get_feature_names_out()]
    class_idx = list(rf.classes_).index(TARGET_CLASS)
    numeric_features = [f for f in (NUMERIC_COLS + DATE_COLS) if f in feature_names]
    num_idx = [feature_names.index(f) for f in numeric_features]

    # remove any figures left behind by older, untagged runs
    for stale in (f"shap_summary_{TARGET_CLASS}.png",
                  f"shap_summary_numeric_{TARGET_CLASS}.png",
                  f"shap_scatter_numeric_{TARGET_CLASS}.png"):
        p = PLOTS_DIR / stale
        if p.exists():
            p.unlink()

    datasets = [("test", X)]
    if TRAIN_PATH.exists():
        datasets.insert(0, ("train", pd.read_csv(TRAIN_PATH)))
    else:
        print(f"note: {TRAIN_PATH.name} not found - re-run train.py to get the train-set plots")
    for tag, X_split in datasets:
        _shap_numeric_plots(X_split, tag, preprocessor, explainer, class_idx,
                            feature_names, numeric_features, num_idx)

    print("\nall figures written to models/plots/")


if __name__ == "__main__":
    main()
