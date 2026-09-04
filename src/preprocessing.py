"""
Reusable preprocessing + model pipeline.

- Col2  -> dropped (near-unique / corrupted identifier, nothing to generalise on)
- Col3  -> numeric, median-impute + robust-scale (heavy skew + outliers)
- Col5  -> a date, so we engineer year / month / quarter and drop the raw stamp
- Col1, Col4, Col6, Col7 -> categorical text, ordinal-encoded (one integer per column).
  Ordinal (not one-hot) keeps the feature space tiny - one column each instead of
  ~2,400 sparse dummies - which a RandomForest handles fine and which makes SHAP
  fast, stable, and naturally one-value-per-column for the explainer.
- the label -> normalised elsewhere (clean_labels) before training
"""

import re

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, RobustScaler

# The raw input schema the client gives us.
RAW_FEATURES = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7"]

# Required columns after preprocessing
# Col2 gone, Col5 expanded
NUMERIC_COLS = ["Col3"]
DATE_COLS = ["Col5_year", "Col5_month", "Col5_quarter"]
CATEGORICAL_COLS = ["Col1", "Col4", "Col6", "Col7"]
PREPARED_COLS = CATEGORICAL_COLS + NUMERIC_COLS + DATE_COLS

# Only these two classes have enough rows to learn/evaluate honestly; everything
# else (Category_3/4/5/6, ~53 rows total, one class has just 2) is folded into
# 'Other'. 
KEEP_CLASSES = {"Category_1", "Category_2"}

# Small, sensible search space for the RandomForest, tuned through the pipeline via
# the 'model__' prefix. Kept modest so it runs fast and is easy to justify: it just
# trades tree depth vs. how aggressively we regularise with leaf size / tree count.
DEFAULT_PARAM_GRID = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [None, 20],
    "model__min_samples_leaf": [1, 3],
}


def clean_label(value):
    """Collapse every messy spelling of a label onto a single canonical Category_N.

    The raw labels are dirty in a few different ways (casing, stray spaces/underscores,
    a missing underscore, and one real typo 'Categry'). We lowercase, strip the
    separators, fix the typo, then just read off the category number.
    """
    s = str(value).strip().lower()
    s = s.replace(" ", "").replace("_", "")
    s = s.replace("categry", "category")  # the one genuine typo
    m = re.search(r"category([0-9]+)", s)
    if m:
        return "Category_" + m.group(1)
    return "Unknown"


# function to clean the label column in a DataFrame
def clean_labels(df, source_col="ClassificationLabel", target_col="label"):
    """Return a copy of df with a cleaned label column added."""
    out = df.copy()
    out[target_col] = out[source_col].apply(clean_label)
    return out


## Function to group rare labels into 'Other' and keep only the frequent ones like Category_1 and Category_2
def group_rare_labels(labels):
    """Fold the ultra-rare classes into a single 'Other' bucket."""
    return labels.where(labels.isin(KEEP_CLASSES), other="Other")


def _prepare_frame(X):
    """Row-level cleaning that must happen the SAME way at train and predict time.

    This includes:
    - Dropping label columns
    - Parsing numeric columns
    - Expanding date columns into year/month/quarter
    - Ensuring categorical columns are strings with NaNs preserved
    - Guaranteeing the final column order matches PREPARED_COLS
    """
    X = pd.DataFrame(X).copy()

    # drop the label/target column
    X = X.drop(columns=[c for c in ("ClassificationLabel", "label") if c in X.columns])

    # Col3: force to a real number; anything unparseable becomes NaN for the imputer
    if "Col3" in X.columns:
        X["Col3"] = pd.to_numeric(X["Col3"], errors="coerce")

    # Col5: extracting year, month, and quarter from the timestamp
    if "Col5" in X.columns:
        d = pd.to_datetime(X["Col5"], errors="coerce")
        X["Col5_year"] = d.dt.year
        X["Col5_month"] = d.dt.month
        X["Col5_quarter"] = d.dt.quarter
        X = X.drop(columns="Col5")
    for c in DATE_COLS:
        if c in X.columns:
            # -1 is an explicit 'unknown date' marker, keeps the column integer-clean
            X[c] = X[c].fillna(-1).astype(int)

    # Col2 is an identifier as we saw in EDA -> drop it entirely 
    if "Col2" in X.columns:
        X = X.drop(columns="Col2")

    # keep every present value a string so the encoder sees consistent types
    for c in CATEGORICAL_COLS:
        if c in X.columns:
            X[c] = X[c].where(X[c].isna(), X[c].astype(str))

    # guarantee the exact column set/order the ColumnTransformer expects
    return X.reindex(columns=PREPARED_COLS)


def build_pipeline(model=None):
    """Assemble the full preprocess -> model pipeline.

    Defaults to a class-weighted RandomForest: it copes with mixed feature types
    and non-linear splits out of the box, is unbothered by Col3's scale/outliers,
    and gives feature_importances_ for free (handy for the explainer later).
    """
    if model is None:
        model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",  # push the model to care about the minority classes
            random_state=42,
            n_jobs=-1,
        )

    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),  # median/IQR scaling shrugs off the 8.8M outliers
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            # one integer per column; unseen categories in an upload map to -1 (safe)
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    features = ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_COLS),
            ("date", "passthrough", DATE_COLS),  # small ints, fine for the tree as-is
            ("cat", categorical, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )

    return Pipeline(
        [
            ("prep", FunctionTransformer(_prepare_frame)),
            ("features", features),
            ("model", model),
        ]
    )


def build_search(model=None, param_grid=None, n_splits=5, random_state=42):
    """Wrap build_pipeline() in a grid search - tuning kept separate from the model.

    build_pipeline stays a plain estimator (that's what we save and what the dashboard
    loads for inference). Tuning is a training-time concern, so it lives in its own
    thin wrapper: one function defines the model, this one defines how we tune it.

    Every candidate is scored with the SAME preprocessing folded inside each CV fold
    (no leakage), and we refit the winner on macro-F1 because accuracy is misleading
    on this imbalanced target.
    """
    pipeline = build_pipeline(model)
    grid = DEFAULT_PARAM_GRID if param_grid is None else param_grid
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return GridSearchCV(
        pipeline,
        grid,
        cv=cv,
        scoring=["accuracy", "balanced_accuracy", "f1_macro", "roc_auc_ovr"],
        refit="f1_macro",  # pick the winner on macro-F1, then refit it on all training data
        n_jobs=-1,
        verbose=1,
    )
