"""
SHAP-based explanations, reused by both evaluate.py (to capture global importance)
and the dashboard's agentic explainer (to explain a single prediction on demand).

Why SHAP + TreeExplainer: the model is a RandomForest, so TreeExplainer gives exact,
fast per-feature contributions with no sampling. Because the categoricals are
ordinal-encoded (one column each), SHAP produces one value per feature directly; the
only folding left is summing the three Col5 date-parts (year/month/quarter) back onto
Col5 so the explanation reads as the 7 original columns a client recognises.
"""

import numpy as np
import pandas as pd
import shap

from src.preprocessing import RAW_FEATURES


def build_tree_explainer(model):
    """Build a SHAP TreeExplainer on the RandomForest step of the pipeline."""
    return shap.TreeExplainer(model.named_steps["model"])


def _feature_to_source(feature_name):
    """Map an encoded feature name back to its raw column.

    ColumnTransformer names look like 'num__Col3', 'date__Col5_year',
    'cat__Col7_NoDoc'. We strip the transformer prefix, then match the raw column.
    """
    name = feature_name.split("__", 1)[-1]  # drop the 'num__' / 'cat__' / 'date__' prefix
    for col in RAW_FEATURES:
        if name == col or name.startswith(col + "_"):
            return col
    return name


def _to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def _shap_for_class(values, class_idx):
    """Slice SHAP values for one class, tolerating both 2-D and 3-D array shapes."""
    if values.ndim == 3:  # (n_samples, n_features, n_classes) for multiclass
        return values[:, :, class_idx]
    return values  # (n_samples, n_features) - single output


def explain_row(row_df, model, preprocessor, explainer, feature_names=None, top_k=6):
    """Explain one prediction: return the predicted class + the top source columns.

    This is the exact call the agent makes for a single uploaded row. It returns the
    contributions already aggregated to the original columns, each with the row's own
    raw value, so the explanation reads in plain business terms.
    """
    if feature_names is None:
        feature_names = preprocessor.named_steps["features"].get_feature_names_out()

    X_dense = _to_dense(preprocessor.transform(row_df))
    pred = model.predict(row_df)[0]
    proba = model.predict_proba(row_df)[0]
    class_idx = list(model.classes_).index(pred)

    shap_row = _shap_for_class(explainer(X_dense).values, class_idx)[0]

    # sum contributions per original column (only Col5's date-parts actually merge)
    contrib = {}
    for fname, sval in zip(feature_names, shap_row):
        src = _feature_to_source(fname)
        contrib[src] = contrib.get(src, 0.0) + float(sval)

    ranked = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    top_features = []
    for src, sval in ranked:
        raw_val = row_df.iloc[0][src] if src in row_df.columns else None
        top_features.append(
            {
                "feature": src,
                "value": None if raw_val is None else str(raw_val),
                "shap": round(sval, 4),
                "direction": "supports" if sval >= 0 else "opposes",
            }
        )

    return {
        "prediction": str(pred),
        "confidence": round(float(proba[class_idx]), 4),
        "top_features": top_features,
    }


def global_importance(model, preprocessor, X, explainer=None):
    """Mean |SHAP| per original column across a dataset - for the Tab 1 overview."""
    if explainer is None:
        explainer = build_tree_explainer(model)
    feature_names = preprocessor.named_steps["features"].get_feature_names_out()

    values = explainer(_to_dense(preprocessor.transform(X))).values
    abs_vals = np.abs(values)
    if abs_vals.ndim == 3:  # average the importance across classes
        abs_vals = abs_vals.mean(axis=2)
    mean_abs = abs_vals.mean(axis=0)  # average across rows -> one score per feature

    agg = {}
    for fname, v in zip(feature_names, mean_abs):
        src = _feature_to_source(fname)
        agg[src] = agg.get(src, 0.0) + float(v)

    return pd.DataFrame(
        sorted(agg.items(), key=lambda kv: kv[1], reverse=True),
        columns=["feature", "mean_abs_shap"],
    )


def rf_importance_by_column(model, preprocessor):
    """Random-forest built-in (impurity) importance, folded to the original columns.

    A cheap sanity-check that sits alongside SHAP: if the two rankings roughly agree,
    we trust the story more. RF importance is model-internal and cardinality-biased,
    SHAP is prediction-level - so they answer slightly different questions.
    """
    feature_names = preprocessor.named_steps["features"].get_feature_names_out()
    importances = model.named_steps["model"].feature_importances_

    agg = {}
    for fname, v in zip(feature_names, importances):
        src = _feature_to_source(fname)
        agg[src] = agg.get(src, 0.0) + float(v)

    return pd.DataFrame(
        sorted(agg.items(), key=lambda kv: kv[1], reverse=True),
        columns=["feature", "rf_importance"],
    )


def shap_table(X, model, preprocessor, explainer=None):
    """Per-row SHAP contributions, folded to the original columns.

    For each row we take the SHAP values of that row's OWN predicted class, so the
    numbers explain the actual prediction (not some fixed reference class). Returns a
    DataFrame aligned to X.index with one 'shap_<col>' per modelled column - the three
    Col5 date-parts are summed back onto Col5, matching the explainer's column view.
    """
    if explainer is None:
        explainer = build_tree_explainer(model)
    feature_names = preprocessor.named_steps["features"].get_feature_names_out()
    sources = [_feature_to_source(f) for f in feature_names]
    ordered_cols = list(dict.fromkeys(sources))  # unique source columns, order preserved

    X_enc = _to_dense(preprocessor.transform(X))
    preds = model.predict(X)
    class_list = list(model.classes_)
    values = explainer(X_enc).values  # (rows, features[, classes])

    records = []
    for i in range(len(X)):
        row_vals = values[i, :, class_list.index(preds[i])] if values.ndim == 3 else values[i]
        agg = {c: 0.0 for c in ordered_cols}
        for src, v in zip(sources, row_vals):
            agg[src] += float(v)
        records.append(agg)

    out = pd.DataFrame(records, index=X.index)
    out.columns = [f"shap_{c}" for c in out.columns]
    return out


