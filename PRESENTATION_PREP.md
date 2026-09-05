# EY Take-Home — Presentation & Interview Prep

A single reference to walk the panel through the project and defend every decision.
Structured around the brief's **presentation flow** and its **four evaluation dimensions**.

---

## 0. 30-second project summary (say this first)

> "The client gave 5,899 labelled rows, Columns A–G as inputs and H as the target. The data was
> messy — a dirty target label, a junk ID column, one heavily-skewed numeric column, a date, and
> several high-cardinality text columns. I cleaned it, built one scikit-learn pipeline that does all
> preprocessing plus a tuned RandomForest, validated it honestly with stratified cross-validation on
> a 90% train split, and reported on an untouched 10% hold-out. I delivered it as a two-tab Streamlit
> dashboard: Tab 1 tells the end-to-end story, Tab 2 takes a live upload, runs the exact same pipeline,
> and lets you click any prediction to get an AI explanation grounded in that row's SHAP values."

---

## 1. Key numbers to memorize (cheat sheet)

**Data**
- 5,899 rows · 7 features (Col1–Col7) · target = ClassificationLabel (Col H)
- Missing values: only **Col4**, 153 rows (~2.6%)
- Col3 (only numeric): median ~1,093, max ~8.8M, min ~−269k, skew ≈ 8.4, 69 negatives
- Col5 (date): 2016-10-01 → 2018-12-31

**Target after cleaning (6 classes, then grouped to 3)**
- Category_1: 5,215 (88.4%) · Category_2: 631 (10.7%) · rest tiny (Category_5 has **2 rows**)
- Grouped for modelling → **Category_1 / Category_2 / Other** (Other = 53 rows, 0.9%)
- **Naïve baseline** (always predict Category_1) = ~88.4% accuracy but macro-F1 ≈ 0.31 → why accuracy alone is a trap

**Split**: stratified 90/10 → **5,309 train / 590 hold-out**; 5-fold stratified CV on the train part.

**Model**: RandomForest (`class_weight='balanced'`), tuned via GridSearchCV.
Best params: `n_estimators=400, max_depth=20, min_samples_leaf=1`.

**Performance**

| Metric | 5-fold CV | Hold-out (10%) |
|---|---|---|
| Accuracy | 0.947 | 0.954 |
| Balanced accuracy | 0.796 | 0.790 |
| Macro-F1 | 0.809 | 0.816 |
| ROC-AUC (OVR) | 0.979 | 0.982 |

**Per-class (hold-out)**: Category_1 F1 ≈ 0.97 · Category_2 F1 ≈ 0.81 · Other F1 ≈ 0.67 (only 5 rows).

**Feature importance** — RF impurity and SHAP **agree** on the ranking:
`Col6 > Col1 > Col3 > Col4 > Col5 > Col7` (Col2 dropped).

---

## 2. Presentation walkthrough (the live 1-hour flow)

### Step 1 — Walk the dashboard (Tab 1)
1. **Exploration**: shape, missing values (only Col4), the **dirty label** (11 spellings of 6 classes),
   the **severe imbalance**, Col3's skew/outliers, records over time, numeric correlation.
   - Punchline: "The single most important finding is the imbalance — it dictates how I evaluate."
2. **Cleaning & Preprocessing**: label normalization before/after table, the per-column decision table,
   and a concrete **one row raw → transformed** view (7 raw cols → 8 model features).
3. **Model & Evaluation**: algorithm, GridSearch best params, CV strategy, the metrics table,
   confusion matrix, classification report, and RF + SHAP importance/summary/scatter plots.
4. **Split strategy**: stratified 90/10, CV as validation, hold-out untouched.

**Interactive controls to show off in Tab 1** (earns the brief's "interactive viz" bonus):
- **Explore any column** — a dropdown over Col1–Col7 that live-shows cardinality, missingness,
  top values, and a distribution chart. Invite the panel to pick a column.
- **Col3 by class** — a class multiselect that overlays density-normalised Col3 distributions,
  answering "does this feature separate the classes?" on demand.
- **SHAP target-class selector** — switch the SHAP beeswarm/scatter between Category_1 / Category_2 /
  Other. SHAP is computed **once and cached** for the whole hold-out (all classes), then just
  **re-sliced** per class — so switching is instant and you can show how the drivers change per class.

### Step 2 — Run the pipeline live (Tab 2)
- Upload `holdout.csv` (Cols A–G). Say: "Same pipeline object that was fit on train — nothing
  pre-computed." Show the 590 predictions + confidence in the scrollable table and the class-mix chart.

### Step 3 — Demo the AI agent
- Click a row → **Explain this prediction** → the LLM returns a plain-English reason built from that
  row's SHAP contributions.
- Click **Show the prompt for this row** → prove the model only receives the real SHAP numbers
  (transparent, not hallucinated). Emphasize: on-demand only, never preloaded.

### Step 4 — Answer questions
- Use the Decision Log (§4) and Q&A (§5).

---

## 3. Mapping to the four evaluation dimensions

### Model Accuracy
- Report **test AND hold-out** numbers (they track closely → not overfit).
- Lead with **macro-F1 / balanced accuracy**, not raw accuracy — explain the 88% baseline trap.
- Honest about the rare class (`Other`, 5 hold-out rows): good but statistically thin.

### Technical Depth
- **Preprocessing**: median impute + RobustScaler for the skewed numeric; explicit `'missing'` token
  for Col4; ordinal encoding with unseen-category handling for uploads.
- **Feature engineering**: dropped the ID column; expanded the date into year/month/quarter.
- **Model selection**: RF with class weighting; GridSearchCV tuned on macro-F1.
- **Validation**: stratified k-fold, untouched hold-out, separate train/evaluate scripts.
- **Pipeline robustness**: one `Pipeline` object fit on train, saved, reused live.
- **Explainability**: SHAP (TreeExplainer) + a deliberate encoding choice that made SHAP stable.

### Dashboard & UX
- Two clear tabs, interactive tables/charts, scrollable predictions.
- **Interactive Tab 1 controls**: an any-column explorer, a per-class Col3 comparison, and a
  live SHAP class selector — so the panel can interrogate the data/model on demand, not just watch.
- **Seamless live pipeline** (single saved object) and a **functional, transparent AI agent**
  (with a prompt-inspector and a SHAP fallback so it never dead-ends).

### Communication
- One consistent story: *dirty data → honest evaluation → explainable model*.
- Every decision has a "why" and an alternative considered (see §4).
- Openly state limitations (§6) — signals maturity.

---

## 4. Decision Log (defend every choice)

| Decision | Why | Alternative considered |
|---|---|---|
| **Drop Col2** | ~4,800 unique, corrupted values (e.g. `4.80Z+11`) → an identifier; memorizing it = leakage/overfit | Engineer format flags (length, has-letters) — low value, skipped |
| **Col3: median impute + RobustScaler** | Heavy right skew + outliers (max 8.8M); median/IQR aren't dragged by extremes | StandardScaler (distorted by outliers); log (breaks on negatives) |
| **Keep outliers** | May be real large amounts; trees are scale-invariant | Winsorize/clip — unnecessary for a tree |
| **Col4 missing → `'missing'` token** | Only 2.6%; "was it blank" may itself be predictive; keeps rare-class rows | Drop rows (loses labelled data); impute mode (invents signal) |
| **Col5 → year/month/quarter, drop raw** | A raw timestamp isn't usable; parts capture seasonality/trend | Leave as datetime (unusable); full date ordinal (less interpretable) |
| **Ordinal encoding for categoricals** | Collapses ~2,400 one-hot dummies → 8 features; makes SHAP **fast, stable, one-value-per-column** | One-hot (SHAP additivity blew up on 2,412 dims); target encoding (leakage risk, more to defend) |
| **Normalize the label** | Same class written 11 ways (casing/spacing/`Categry` typo) — model would split them | Leave as-is (silently wrong) |
| **Group rare classes → `Other`** | Category_5 has 2 rows — impossible to split/evaluate honestly | Keep 6 classes (can't stratify/validate); drop them (loses info) |
| **RandomForest + `class_weight='balanced'`** | Mixed features, non-linear, robust to scale; weighting fights imbalance; free feature importances | LogisticRegression baseline (linear, weaker); XGBoost (more tuning to justify) |
| **GridSearchCV, refit on macro-F1** | Optimize the metric that reflects minority-class skill, not the inflated accuracy | Optimize accuracy (would ignore minorities) |
| **Stratified 90/10 + stratified 5-fold CV** | Keeps class proportions in every split — essential under imbalance | Random split (rare classes could vanish from a fold) |
| **One Pipeline object, fit on train, saved** | Guarantees Tab 2 applies identical transforms with no manual steps; no train/test skew | Manual re-preprocessing in the app (fragile, leakage-prone) |
| **Separate train.py / evaluate.py** | Mirrors deployment: train produces artifacts, evaluation only consumes them | One script (blurs the train/test boundary) |
| **SHAP TreeExplainer** | Exact, fast per-feature attributions for tree models; grounds the agent | KernelSHAP (slow, approximate) |
| **LLM agent + SHAP fallback** | On-demand, transparent explanations; fallback keeps the demo alive if the gateway fails | LLM-only (a network hiccup breaks the live demo) |

---

## 5. Anticipated Q&A

**Why is accuracy misleading here?**
Predicting "always Category_1" already scores ~88%. So I judge on macro-F1 (0.82) and balanced
accuracy (0.79), which weight the minority classes, plus the confusion matrix.

**Did you do EDA on all the data or just train? Isn't that leakage?**
Exploration for *understanding* and *data-quality* (shapes, missing, dirty labels, imbalance) is on
the full set — none of it is fitted. Every *learned* parameter (imputation medians, scaler stats,
encoder categories) is fit **inside the pipeline on the training split only**, so the hold-out stays
truly unseen. Label cleaning is deterministic string normalization, not learned, so it's safe globally.

**Where's the validation set?** The 5-fold stratified CV on the 90% train part *is* the validation —
every fold acts as validation once. The 10% hold-out is the final, untouched test.

**Why ordinal encoding for nominal text — isn't the order arbitrary?**
For a tree it's fine: trees threshold-split, so an arbitrary code order still lets them isolate
categories. I accepted a small accuracy cost (~few points macro-F1 vs one-hot) to get **stable,
column-level SHAP** and an 8-feature space instead of 2,400. Explainability is a graded deliverable,
so that trade was worth it. If I needed the accuracy back, target encoding would be the next step.

**Why not one-hot?** With Col4 at ~2,000 categories, one-hot makes thousands of sparse, near-useless
columns, and SHAP's additivity became numerically unstable on that 2,412-dim space. Ordinal fixed both.

**How does Tab 2 guarantee the same preprocessing?** It loads the single fitted `Pipeline` object and
calls `.predict()`. The imputers/scaler/encoder were fit on train and are frozen inside it;
`handle_unknown` maps unseen categories to −1 so an upload can't crash it.

**How does the agent explanation work / can it hallucinate?**
On click, I compute SHAP for that one row, fold it to the 7 original columns, and send *only* those
numbers to the LLM with a system prompt that forbids inventing features. The "Show the prompt" button
proves exactly what's sent. If the LLM is unavailable, a deterministic SHAP sentence is shown instead.

**Is the model overfitting?** CV and hold-out metrics are within ~1–2 points, and the SHAP train vs
test plots show the same feature effects — both are evidence it generalizes.

**What are RF importance vs SHAP telling you?** They independently agree on the ranking
(Col6 > Col1 > Col3 > Col4 > Col5 > Col7), which builds trust. RF importance is impurity-based and
cardinality-biased; SHAP is prediction-level and signed.

**Are the Tab 1 SHAP plots pre-computed or generated live?** Generated live, but SHAP itself runs
**once per session and is cached** — it produces a `(rows, features, classes)` array for the whole
hold-out; the class selector just **re-slices** that array and redraws the figures. So no static
per-class images: the class switch is an instant re-slice, always in sync with the loaded model.
The only static PNGs are the two *global* importance charts.

**Why RandomForest over deep learning / boosting?** 5,899 rows with mixed messy features — RF is
strong out-of-the-box, robust to scale/outliers, interpretable, and fast to tune. A neural net is
overkill and harder to justify; XGBoost is a reasonable next step but more tuning to defend.

**How would you improve it with more time?** Target encoding for the high-cardinality columns,
threshold tuning per class, calibrated probabilities, and collecting more rare-class examples.

---

## 6. Known limitations (say these before they ask)
- **Rare classes are under-supported** — `Other` has 5 hold-out rows, so its metrics are indicative,
  not reliable. Category_5 (2 rows) can't be modelled on its own.
- **Ordinal encoding trades a little accuracy for explainability** — a conscious choice, reversible.
- **Col2 discarded** — if it secretly encoded a real signal we'd lose it, but its corruption/uniqueness
  makes that unlikely and risky to use.
- **PDP/SHAP colour for categorical codes isn't semantically ordered** — I only trust magnitude there,
  which is why the SHAP summary/scatter are restricted to the genuinely numeric features.

---

## 7. Architecture / file map
- `src/preprocessing.py` — the pipeline: cleaning, `_prepare_frame`, `build_pipeline`, `build_search`
- `src/train.py` — clean → stratified split → GridSearch → save model + fitted preprocessor + metrics
- `src/evaluate.py` — load artifacts → score hold-out → per-row results + SHAP
- `src/explain.py` — SHAP helpers (TreeExplainer, `explain_row`, importance, `shap_table`)
- `src/plots.py` — RF/SHAP importance + SHAP summary/scatter (train & test)
- `src/agent.py` — on-demand LLM explainer (Azure gateway) + SHAP fallback + prompt builder
- `app/streamlit_app.py` — the two-tab dashboard
- `eda.ipynb` — the exploration narrative

---

## 8. Live demo runbook
```bash
source .venv/bin/activate
python src/train.py       # -> models/model.joblib, preprocessor.joblib, metrics.json, train.csv, holdout.csv
python src/evaluate.py    # -> confusion_matrix.csv, holdout_metrics.json, holdout_predictions.csv
python src/plots.py       # -> models/plots/*.png
streamlit run app/streamlit_app.py
```
Demo order: Tab 1 top-to-bottom → Tab 2 upload `holdout.csv` → select a row → Explain → Show prompt.

> ⚠️ Before sending to EY: remove the live secret from `.env` (it's an internal credential) and either
> include the `models/` folder or tell the grader to run the three scripts above first.
