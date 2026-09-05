"""
EY take-home dashboard.

Run from the project root (after train.py / evaluate.py / plots.py):
    streamlit run app/streamlit_app.py

Tab 1 - Exploration, Preprocessing & Model Results (the end-to-end story).
Tab 2 - Hold-out upload -> live prediction pipeline -> per-row agentic explanation.

The app never retrains: it loads the frozen artifacts produced by the src/ scripts.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so src.* is importable and the pickle can re-hydrate

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.metrics import classification_report

import src.preprocessing  # noqa: F401  (registers module path for the pickled transformer)
from src import agent
from src.explain import build_tree_explainer, explain_row
from src.preprocessing import (
    DATE_COLS,
    NUMERIC_COLS,
    RAW_FEATURES,
    clean_label,
    group_rare_labels,
)

MODELS_DIR = ROOT / "models"
PLOTS_DIR = MODELS_DIR / "plots"
DATA_PATH = ROOT / "Challenge_Data.xlsx"
HOLDOUT_PATH = ROOT / "holdout.csv"

st.set_page_config(page_title="EY Data Science Challenge", layout="wide")


# --------------------------------------------------------------------------- #
# cached loaders - artifacts are read once and reused across reruns
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_model():
    model = joblib.load(MODELS_DIR / "model.joblib")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    explainer = build_tree_explainer(model)
    return model, preprocessor, explainer


@st.cache_data
def load_json(name):
    import json

    with open(MODELS_DIR / name) as f:
        return json.load(f)


@st.cache_data
def load_csv(path, **kw):
    return pd.read_csv(path, **kw)


@st.cache_data
def load_raw():
    df = pd.read_excel(DATA_PATH)
    df["label_clean"] = df["ClassificationLabel"].apply(clean_label)
    df["label_grouped"] = group_rare_labels(df["label_clean"])  # 3-class view used for comparisons
    return df


@st.cache_data
def compute_holdout_shap():
    """Compute SHAP once on the hold-out; the class selector just re-slices this."""
    model, preprocessor, explainer = load_model()
    X = pd.read_csv(HOLDOUT_PATH)
    feat_names = [n.split("__", 1)[-1] for n in preprocessor.named_steps["features"].get_feature_names_out()]
    shap_vals = explainer(np.asarray(preprocessor.transform(X))).values  # (rows, features[, classes])
    numeric_features = [f for f in (NUMERIC_COLS + DATE_COLS) if f in feat_names]
    num_idx = [feat_names.index(f) for f in numeric_features]
    # unscaled values so the axes read in real units (Col3 amount, months 1-12)
    X_display = src.preprocessing._prepare_frame(X)[numeric_features].reset_index(drop=True)
    return shap_vals, numeric_features, num_idx, X_display, list(model.classes_)


def artifacts_ready():
    return (MODELS_DIR / "model.joblib").exists()


# --------------------------------------------------------------------------- #
# TAB 1
# --------------------------------------------------------------------------- #
def render_exploration(raw):
    st.header("1. Data Exploration")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{raw.shape[0]:,}")
    c2.metric("Feature columns (A-G)", "7")
    c3.metric("Target classes (raw)", raw["ClassificationLabel"].nunique())
    c4.metric("Missing cells", int(raw[RAW_FEATURES].isna().sum().sum()))

    st.subheader("Raw sample")
    st.dataframe(raw[RAW_FEATURES + ["ClassificationLabel"]].head(8), use_container_width=True)

    st.subheader("Explore any column")
    col = st.selectbox("Pick a feature column to drill into", RAW_FEATURES, key="explore_col")
    s = raw[col]
    e1, e2, e3 = st.columns(3)
    e1.metric("Unique values", f"{s.nunique():,}")
    e2.metric("Missing", f"{int(s.isna().sum()):,}")
    e3.metric("Dtype", str(s.dtype))
    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown("**Top values**")
        vc = s.value_counts(dropna=False).head(15).rename_axis(col).reset_index(name="count")
        st.dataframe(vc, use_container_width=True, hide_index=True)
    with ec2:
        st.markdown("**Distribution**")
        if pd.api.types.is_numeric_dtype(s):
            lo_e, hi_e = s.quantile([0.01, 0.99])
            clip = s[(s >= lo_e) & (s <= hi_e)].dropna()
            fige, axe = plt.subplots(figsize=(5, 3.5))
            axe.hist(clip, bins=40, color="steelblue", edgecolor="white")
            axe.set_title(f"{col} (clipped 1-99 pct)")
            axe.set_xlabel(f"{col} value")
            axe.set_ylabel("count")
            fige.tight_layout()
            st.pyplot(fige)
        elif pd.api.types.is_datetime64_any_dtype(s):
            st.bar_chart(s.dt.to_period("M").astype(str).value_counts().sort_index())
        else:
            st.bar_chart(s.value_counts().head(15))

    left, right = st.columns(2)
    with left:
        st.subheader("Missing values")
        miss = raw[RAW_FEATURES].isna().sum()
        miss_df = pd.DataFrame(
            {"missing": miss, "missing_%": (miss / len(raw) * 100).round(2)}
        )
        st.dataframe(miss_df, use_container_width=True)
        st.caption("Only Col4 has gaps (~2.6%). We keep those rows and fill with a 'missing' token.")
    with right:
        st.subheader("Col3 - the one numeric feature")
        desc = raw["Col3"].describe()[["mean", "std", "min", "50%", "max"]].round(1)
        st.dataframe(desc.to_frame("Col3"), use_container_width=True)
        st.caption("Median ~1k but max ~8.8M and some negatives -> heavy skew + outliers.")

    st.subheader("Target label distribution (the headline finding)")
    lc, rc = st.columns(2)
    with lc:
        st.markdown("**Raw (dirty) values**")
        st.dataframe(raw["ClassificationLabel"].value_counts().to_frame("count"), use_container_width=True)
    with rc:
        st.markdown("**After cleaning the spelling variants**")
        st.bar_chart(raw["label_clean"].value_counts())
    st.info(
        "Severely imbalanced: Category_1 ~88%, Category_2 ~11%, the rest tiny (Category_5 has 2 rows). "
        "So accuracy alone is misleading - we judge the model on macro-F1 / balanced accuracy."
    )

    st.subheader("Col3 distribution & outliers")
    lo, hi = raw["Col3"].quantile([0.01, 0.99])
    clipped = raw["Col3"][(raw["Col3"] >= lo) & (raw["Col3"] <= hi)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
    ax1.hist(clipped, bins=50, color="steelblue", edgecolor="white")
    ax1.set_title("Col3 (clipped 1-99 pct)")
    ax1.set_xlabel("Col3 value")
    ax1.set_ylabel("count")
    ax2.boxplot(raw["Col3"].dropna())
    ax2.set_title("Col3 with outliers")
    ax2.set_ylabel("Col3 value")
    ax2.set_xticklabels(["Col3"])
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Col3 by class - does it separate the classes?")
    classes_all = sorted(raw["label_grouped"].unique())
    picked = st.multiselect("Classes to overlay", classes_all, default=classes_all, key="col3_classes")
    if picked:
        lo_c, hi_c = raw["Col3"].quantile([0.01, 0.99])
        figb, axb = plt.subplots(figsize=(9, 3.8))
        for c in picked:
            vals = raw.loc[(raw["label_grouped"] == c) & raw["Col3"].between(lo_c, hi_c), "Col3"].dropna()
            if len(vals):
                axb.hist(vals, bins=40, alpha=0.5, density=True, label=f"{c} (n={len(vals)})")
        axb.set_xlabel("Col3 (clipped 1-99 pct)")
        axb.set_ylabel("density")
        axb.legend()
        figb.tight_layout()
        st.pyplot(figb)
        st.caption("Density-normalised so classes of very different sizes are comparable; separation here means Col3 is informative.")
    else:
        st.info("Pick at least one class to compare.")

    st.subheader("Records over time (Col5)")
    by_month = raw.groupby(pd.to_datetime(raw["Col5"]).dt.to_period("M").astype(str)).size()
    st.line_chart(by_month)

    st.subheader("Correlation (numeric features)")
    num = raw[["Col3"]].copy()
    d = pd.to_datetime(raw["Col5"])
    num["Col5_year"], num["Col5_month"], num["Col5_quarter"] = d.dt.year, d.dt.month, d.dt.quarter
    corr = num.corr()
    figc, axc = plt.subplots(figsize=(4.5, 3.5))
    im = axc.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    axc.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    axc.set_yticks(range(len(corr.index)), corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            axc.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    figc.colorbar(im, fraction=0.046)
    figc.tight_layout()
    st.pyplot(figc)
    st.caption(
        "Only Col3 is a genuine numeric feature (the rest are engineered date parts), so linear "
        "correlation is limited - for the categorical text columns we rely on SHAP / feature "
        "importance (in the Model section) to judge feature-target relationships."
    )



def render_preprocessing(raw):
    st.header("2. Data Cleaning & Preprocessing")

    st.subheader("Label cleaning - before / after")
    mapping = (
        raw.groupby("ClassificationLabel")
        .agg(mapped_to=("label_clean", "first"), rows=("label_clean", "size"))
        .sort_values("rows", ascending=False)
    )
    st.dataframe(mapping, use_container_width=True)
    st.caption("Lowercase, strip spaces/underscores, fix the 'Categry' typo, then read the category number.")

    st.subheader("Per-column decisions")
    decisions = pd.DataFrame(
        [
            ["Col1", "categorical text (313)", "none", "ordinal-encode"],
            ["Col2", "near-unique / corrupted ID", "none", "DROP (identifier)"],
            ["Col3", "numeric, skewed, outliers", "none", "median impute + RobustScaler"],
            ["Col4", "categorical text (2000)", "153", "fill 'missing' + ordinal-encode"],
            ["Col5", "date", "none", "engineer year / month / quarter, drop raw"],
            ["Col6", "categorical text (200)", "none", "ordinal-encode"],
            ["Col7", "clean categorical (4)", "none", "ordinal-encode"],
            ["label (H)", "target, 6 classes", "none", "normalise; rare -> 'Other'"],
        ],
        columns=["column", "what it is", "missing", "decision"],
    )
    st.dataframe(decisions, use_container_width=True, hide_index=True)

    st.subheader("One row: raw -> transformed")
    model, preprocessor, _ = load_model()
    sample = raw[RAW_FEATURES].head(1)
    feat_names = [n.split("__", 1)[-1] for n in preprocessor.named_steps["features"].get_feature_names_out()]
    after = pd.DataFrame(np.asarray(preprocessor.transform(sample)), columns=feat_names).round(3)
    st.markdown("**Before (raw Cols A-G):**")
    st.dataframe(sample, use_container_width=True)
    st.markdown("**After (what the model sees - Col2 dropped, Col5 expanded, categoricals encoded):**")
    st.dataframe(after, use_container_width=True)


def render_split():
    st.header("3. Split Strategy")
    meta = load_json("metrics.json")
    c1, c2, c3 = st.columns(3)
    c1.metric("Train rows (90%)", f"{meta['n_train']:,}")
    c2.metric("Hold-out rows (10%)", f"{meta['n_holdout']:,}")
    c3.metric("CV folds", "5 (stratified)")
    st.markdown(
        """
- **Stratified 90 / 10 split** - stratified so the rare classes keep their proportion in both parts.
- The **10% hold-out is set aside and never touched** during training; it's what Tab 2 scores live.
- On the 90% we run **5-fold stratified cross-validation** for model selection / tuning.
- We report the hold-out numbers as the honest, out-of-sample estimate.
        """
    )


def render_model():
    st.header("4. Model Training & Evaluation")
    meta = load_json("metrics.json")
    holdout = load_json("holdout_metrics.json")

    st.subheader("Algorithm & tuning")
    st.markdown(
        "- **RandomForest** (`class_weight='balanced'`) - handles mixed features, outliers, non-linearity; "
        "gives feature importances for free.\n"
        "- **GridSearchCV** over `n_estimators`, `max_depth`, `min_samples_leaf`, refit on **macro-F1**."
    )
    st.write("Best hyper-parameters:", meta["best_params"])

    st.subheader("Performance")
    cv = meta["cv_means"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy (hold-out)", f"{holdout['accuracy']:.3f}", f"CV {cv['accuracy']:.3f}")
    m2.metric("Balanced accuracy", f"{holdout['balanced_accuracy']:.3f}", f"CV {cv['balanced_accuracy']:.3f}")
    m3.metric("Macro-F1", f"{holdout['f1_macro']:.3f}", f"CV {cv['f1_macro']:.3f}")
    m4.metric("ROC-AUC (OVR)", f"{holdout['roc_auc_ovr']:.3f}", f"CV {cv['roc_auc_ovr']:.3f}")
    st.caption("Accuracy ~0.96 but macro-F1 ~0.82 - the gap is exactly why we don't trust accuracy alone here.")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix (hold-out)")
        cm = load_csv(MODELS_DIR / "confusion_matrix.csv", index_col=0)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(cm.values, cmap="Blues")
        ax.set_xticks(range(len(cm.columns)), cm.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(cm.index)), cm.index)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm.values[i, j], ha="center",
                        color="white" if cm.values[i, j] > cm.values.max() / 2 else "black")
        fig.colorbar(im, fraction=0.046)
        fig.tight_layout()
        st.pyplot(fig)
    with right:
        st.subheader("Classification report (hold-out)")
        preds = load_csv(MODELS_DIR / "holdout_predictions.csv")
        rep = classification_report(preds["actual_label"], preds["predicted_label"],
                                    output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).transpose().round(3), use_container_width=True)

    st.subheader("Feature importance & SHAP")
    g1, g2 = st.columns(2)
    _img(g1, "rf_feature_importance.png", "RandomForest importance")
    _img(g2, "shap_global_importance.png", "SHAP global importance")
    st.caption("Two independent methods agree: Col6 and Col1 are the dominant drivers.")

    st.markdown("**SHAP for the numeric features - pick which class to explain:**")
    if HOLDOUT_PATH.exists():
        shap_vals, numeric_features, num_idx, X_display, classes = compute_holdout_shap()
        target = st.selectbox(
            "SHAP target class", classes,
            index=classes.index("Category_2") if "Category_2" in classes else 0,
            key="shap_class",
        )
        ci = classes.index(target)
        shap_cls = shap_vals[:, :, ci] if shap_vals.ndim == 3 else shap_vals
        sc1, sc2 = st.columns(2)
        with sc1:
            plt.figure()
            shap.summary_plot(shap_cls[:, num_idx], X_display, show=False)
            fig_bee = plt.gcf()
            fig_bee.suptitle(f"SHAP summary - {target}", y=1.02)
            st.pyplot(fig_bee)
            plt.close(fig_bee)
        with sc2:
            n = len(num_idx)
            nrows = int(np.ceil(n / 2))
            fig_sc, axes = plt.subplots(nrows, 2, figsize=(8, 3.2 * nrows))
            axes = np.atleast_1d(axes).ravel()
            for ax, i, name in zip(axes, num_idx, numeric_features):
                xv = X_display[name].values
                ax.scatter(xv, shap_cls[:, i], s=10, alpha=0.5, color="steelblue")
                ax.axhline(0, color="grey", lw=0.8)
                ax.set_title(name)
                ax.set_xlabel("value")
                ax.set_ylabel("SHAP")
                lo_s, hi_s = np.nanpercentile(xv, [1, 99])
                if hi_s > lo_s:
                    ax.set_xlim(lo_s, hi_s)
            for ax in axes[n:]:
                ax.set_visible(False)
            fig_sc.suptitle(f"SHAP vs value - {target}")
            fig_sc.tight_layout()
            st.pyplot(fig_sc)
            plt.close(fig_sc)
        st.caption("Numeric features only - the beeswarm colour (feature value) is meaningful for these, "
                   "but not for the ordinal-encoded text columns.")
    else:
        st.info("holdout.csv not found - run `python src/train.py` first.")


def _img(col, name, caption):
    p = PLOTS_DIR / name
    if p.exists():
        col.image(str(p), caption=caption, use_container_width=True)
    else:
        col.info(f"{name} not found - run `python src/plots.py`")


# --------------------------------------------------------------------------- #
# TAB 2
# --------------------------------------------------------------------------- #
def render_prediction():
    st.header("Hold-Out Set Prediction Pipeline")
    st.markdown(
        "Upload a hold-out file with **Columns A-G (Col1..Col7)**. The full preprocessing pipeline "
        "runs automatically and predictions are produced live - nothing is pre-computed."
    )
    model, preprocessor, explainer = load_model()

    uploaded = st.file_uploader("Upload hold-out CSV (Cols A-G)", type=["csv"])
    if uploaded is None:
        st.caption("Tip: use the `holdout.csv` produced by train.py for the demo.")
        return

    try:
        data = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return

    missing_cols = [c for c in RAW_FEATURES if c not in data.columns]
    if missing_cols:
        st.error(f"Uploaded file is missing required columns: {missing_cols}")
        return

    X = data[RAW_FEATURES].copy()
    with st.spinner("Running preprocessing + inference..."):
        preds = model.predict(X)
        proba = model.predict_proba(X).max(axis=1)

    results = X.copy()
    results.insert(0, "row", range(len(results)))
    results["prediction"] = preds
    results["confidence"] = proba.round(3)

    st.success(f"Predicted {len(results):,} rows live.")
    st.bar_chart(pd.Series(preds).value_counts())

    st.subheader("Predictions")
    st.caption("Scroll the table and click a row to select it, then explain that prediction.")
    event = st.dataframe(
        results,
        use_container_width=True,
        height=780,  # ~20-30 rows visible, scrollable
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="pred_table",
    )

    st.subheader("Agentic Prediction Explainer")
    selected = event.selection.rows if event and event.selection else []
    if not selected:
        st.info("Select a row above to enable the Explain action.")
        return

    idx = selected[0]
    st.write(f"Selected **row {idx}** -> predicted **{results.iloc[idx]['prediction']}** "
             f"(confidence {results.iloc[idx]['confidence']:.0%})")

    # compute (and cache) the SHAP explanation dict for this row once; both buttons reuse it
    def row_explanation(i):
        store = st.session_state.setdefault("row_expl", {})
        if i not in store:
            store[i] = explain_row(X.iloc[[i]], model, preprocessor, explainer)
        return store[i]

    b1, b2 = st.columns(2)
    if b1.button("Explain this prediction", type="primary"):
        with st.spinner("Asking the agent..."):
            st.session_state.setdefault("explanations", {})[idx] = agent.explain_prediction(row_explanation(idx))
    if b2.button("Show the prompt for this row"):
        st.session_state.setdefault("show_prompt", {})[idx] = True

    # the agent's answer (if already requested for this row)
    text = st.session_state.get("explanations", {}).get(idx)
    if text:
        st.markdown(text)
        st.markdown("**Top feature contributions (SHAP for the predicted class):**")
        st.dataframe(pd.DataFrame(row_explanation(idx)["top_features"]),
                     use_container_width=True, hide_index=True)

    # the exact system + user prompt generated for this row (on demand)
    if st.session_state.get("show_prompt", {}).get(idx):
        system_prompt, user_prompt = agent.build_messages(row_explanation(idx))
        with st.expander("Prompt sent to the agent for this row", expanded=True):
            st.markdown("**System prompt** (fixed rules)")
            st.code(system_prompt, language="text")
            st.markdown("**User prompt** (built dynamically from this row's SHAP values)")
            st.code(user_prompt, language="text")


# --------------------------------------------------------------------------- #
def main():
    st.title("EY Data Science Take-Home Challenge")
    if not artifacts_ready():
        st.error("Model artifacts not found. Run `python src/train.py` (then evaluate.py, plots.py) first.")
        return

    tab1, tab2 = st.tabs(
        ["Tab 1 - Exploration, Preprocessing & Results", "Tab 2 - Hold-Out Prediction + Explainer"]
    )
    with tab1:
        raw = load_raw()
        render_exploration(raw)
        st.divider()
        render_preprocessing(raw)
        st.divider()
        render_split()
        st.divider()
        render_model()
    with tab2:
        render_prediction()


if __name__ == "__main__":
    main()
