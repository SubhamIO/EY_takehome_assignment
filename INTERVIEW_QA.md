# EY Take-Home — Interview Q&A Bank

Intuitive, explain-it-like-you-mean-it answers to the questions most likely to come up.
Grounded in the actual project numbers. Pair this with `PRESENTATION_PREP.md` (the decision log + cheat sheet).

**One-line anchors to keep in your head**
- Data: 5,899 rows · 7 features (Col1–Col7) · target Col H.
- Target after cleaning + grouping: **Category_1 88.4% / Category_2 10.7% / Other 0.9%** (3 classes).
- Split: stratified **90/10 → 5,309 train / 590 hold-out**, 5-fold stratified CV on train.
- Model: **RandomForest** (`class_weight='balanced'`), tuned → `n_estimators=400, max_depth=20, min_samples_leaf=1`.
- Hold-out: **accuracy 0.954 · balanced-acc 0.790 · macro-F1 0.816 · ROC-AUC 0.982**.
- Naïve "always Category_1" baseline: **accuracy 0.88 but macro-F1 ≈ 0.31**.

---

## A. Data & EDA

**Q: Walk me through what you found in the data. What surprised you?**
The headline finding is a **severely imbalanced target** with a **dirty label column**, sitting on top of otherwise mixed, messy features. Concretely: the label is written 11 different ways for what are really 6 classes; one column (Col2) is a corrupted near-unique identifier; one column (Col3) is genuinely numeric but wildly skewed with outliers up to 8.8M; one is a date (Col5); and the rest are high-cardinality text. The surprise was how much of the work was **data quality**, not modelling — if you don't clean the label first, everything downstream is quietly wrong.

**Q: Which columns are categorical vs numerical, and how did you decide?**
I decided from *what's actually inside each column*, not the dtype pandas guessed. Only **Col3** is a real number (it parses cleanly and has a continuous range). **Col5** is a date. **Col1, Col4, Col6, Col7** are categorical text ("WordN" tokens / short codes). **Col2** looks numeric-ish but is a corrupted identifier, so I treat it as neither and drop it. The tell for "categorical" is a bounded, repeating set of values; the tell for "identifier" is near-unique-per-row.

**Q: How did you handle missing values, and why keep the 153 Col4 rows?**
Only Col4 has gaps — 153 rows (~2.6%). I **kept** them and fill with an explicit `'missing'` token rather than dropping, for two reasons: (1) with classes this rare, I don't want to throw away any labelled rows I don't have to; (2) Col4 is categorical, so `'missing'` simply becomes its own category — and "this field was blank" might itself be predictive. Dropping would lose data; imputing a fake real value would invent signal.

**Q: Col3 goes up to 8.8M and has negatives — real or error?**
I treated them as **real but extreme**. The distribution is heavily right-skewed (skew ≈ 8.4): median ~1,000 but a long tail to millions, plus ~69 negative values. Nothing in the brief says they're errors, and amounts legitimately can be large or negative (refunds/adjustments). So I don't delete them — I use a **RobustScaler** (median/IQR) so the extremes don't dominate, and I lean on a tree model that's insensitive to scale.

**Q: Did you check correlations? What did they tell you?**
Yes, but their value is limited here: only Col3 (plus the engineered date parts) is numeric, so a Pearson correlation matrix only covers those. It showed nothing strong. That's *expected* — most of the signal lives in the categorical text columns, where linear correlation doesn't apply. So for feature-vs-target relationships I rely on **SHAP and feature importance** instead, which is the honest tool for this feature mix.

**Q: Did you do EDA on all the data — isn't that leakage?**
No leakage, because I separate **looking** from **fitting**. Exploration for *understanding* and *data quality* (shapes, missingness, dirty labels, imbalance) is fine on the full set — none of those observations are a learned parameter. Every parameter that *is* learned (imputation medians, the scaler's median/IQR, the encoder's category list) is fit **inside the pipeline on the training split only**, so the hold-out stays genuinely unseen. Label cleaning is deterministic string normalization, not something learned from the data, so applying it globally is safe too.

---

## B. Target label

**Q: The label had 11 spellings — how did you find it, and how do you trust your cleaning?**
I did a `value_counts()` on the raw label and saw the same class written as `Category_1`, `category_1`, `Category 3`, `Category _3`, `Category4`, and even a typo `Categry_6`. My cleaner lowercases, strips spaces and underscores, fixes the `categry`→`category` typo, then extracts the trailing digit → `Category_<n>`. I **verified** it by mapping every raw value to its cleaned value and checking that (a) the row counts still add up to 5,899 and (b) *zero* rows fell through to "Unknown". If any variant hadn't been caught, that count would be non-zero.

**Q: Why merge the rare classes into "Other"?**
After cleaning it's a 6-class problem, but the tail is tiny: Category_3 (23), Category_4 (16), Category_6 (12), and **Category_5 has just 2 rows**. You cannot stratify 2 rows across train/val/test, and any per-class metric on them is meaningless. Grouping them into `Other` (53 rows total, 0.9%) gives a clean, *evaluable* 3-class problem and an honest story. I keep the mapping visible so nothing is hidden — and I'm upfront that even `Other` is statistically thin.

**Q: Isn't merging throwing away signal?**
It merges *labels*, not *features* — every row is still there with all its features. The trade is: I lose the ability to distinguish, say, Category_3 from Category_6, but I gain a model I can actually validate. With 2–23 rows per rare class, distinguishing them reliably isn't achievable anyway; pretending otherwise would be the real mistake.

---

## C. Preprocessing & feature engineering

**Q: Why drop Col2?**
Col2 has ~4,800 distinct values across 5,899 rows — nearly unique per row — and it's a mix of numbers, codes, and clearly corrupted values like `4.80Z+11` (a number mangled on export). That's the fingerprint of an **identifier**, not a feature. An almost-unique ID gives the model nothing to generalize from; worst case it memorizes row IDs (leakage/overfit). If pushed, I could engineer cheap format flags (length, has-letters), but the value is low and I'd rather keep the pipeline clean.

**Q: Why RobustScaler and not StandardScaler or log?**
`RobustScaler` centres on the median and scales by the IQR, so the 8.8M outliers don't dominate. A plain `StandardScaler` uses mean/std, which those extremes *distort* — the "typical" value would get squashed near zero. A log transform would tame the skew but **breaks on the ~69 negative values**. Honestly, for the tree model scaling is largely cosmetic (trees split on thresholds, not distances), but I keep robust scaling so the pipeline also behaves well for any linear baseline.

**Q: Why ordinal-encode the text columns instead of one-hot? Isn't imposing an order on nominal data wrong?**
Two reasons drove it. First, cardinality: Col4 alone has ~2,000 categories — one-hot would explode to ~2,400 sparse columns that trees rarely split on usefully. Ordinal keeps it to **one integer per column → 8 features total**. Second, and decisively: with one-hot, my **SHAP additivity became numerically unstable** on that ~2,400-dim space; ordinal made SHAP fast, stable, and naturally one-value-per-column. On the "arbitrary order" worry — for a **tree** it's fine, because trees threshold-split and can still isolate any category with enough splits. I accept a small accuracy cost (a few macro-F1 points vs one-hot) in exchange for stable explainability, which is a graded deliverable. If I needed the accuracy back, target encoding would be the next step.

**Q: What if Tab 2 gets a category never seen in training?**
Handled by design. The encoder is `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)`, so any unseen category in an upload maps to **−1** instead of crashing. The tree treats −1 as just another value. That's what makes the upload pipeline robust "without manual intervention," as the brief requires.

**Q: Why engineer year/month/quarter from the date?**
A raw timestamp is useless to a classifier as-is — it can't compare `2018-05-31` meaningfully. The signal in a date lives in its *parts*: year (trend), month/quarter (seasonality). So I extract those three and drop the raw stamp. It's the one column that genuinely benefits from feature engineering.

**Q: Does scaling even matter for a RandomForest?**
Not really — and that's the honest answer. Trees split on "is feature ≤ threshold", which is invariant to monotonic scaling, so RobustScaler doesn't change the RF's decisions. I keep it because it's harmless, it future-proofs the pipeline for a linear model, and it makes the "one row raw → transformed" view in the dashboard clearer. I'd rather you ask me this than assume I didn't know it.

---

## D. Model selection & training

**Q: Why RandomForest? Why not logistic regression, XGBoost, or a neural net?**
For 5,899 rows of mixed, messy, mostly-categorical features, RandomForest is the pragmatic sweet spot: it handles mixed feature types and non-linear interactions out of the box, is unbothered by Col3's scale and outliers, needs little tuning, resists overfitting via bagging, and gives `feature_importances_` for free (useful for the explainer). Logistic regression is a fine *interpretable baseline* but weaker on non-linear structure. XGBoost is a reasonable next step but adds tuning surface I'd have to justify. A neural net is overkill at this data size and much harder to explain — which cuts against a graded "explainability" deliverable.

**Q: What does `class_weight='balanced'` do?**
It tells the model that mistakes on rare classes should hurt more, by weighting each class inversely to its frequency. Without it, the model can score ~88% by ignoring the minorities entirely; with it, the loss "cares" about Category_2 and Other, so the decision boundary shifts to actually catch them. It's a cost-based fix for imbalance that needs no resampling.

**Q: Did you compare against a baseline?**
Yes — the **naïve majority baseline** ("always predict Category_1") is the reference: it gets ~88% accuracy but a macro-F1 of only ~0.31 and balanced accuracy 0.33. My model's macro-F1 is 0.82 and balanced accuracy 0.79 — that gap is the real measure of skill, and it's the reason accuracy alone is misleading here.

---

## E. Tuning & validation

**Q: Describe your hyperparameter tuning.**
`GridSearchCV` over the RandomForest, tuned **through the pipeline** (via the `model__` prefix) so every candidate is scored with the *same* preprocessing folded inside each CV fold — no leakage. Grid: `n_estimators ∈ {200,400}`, `max_depth ∈ {4,6,12,20,None}`, `min_samples_leaf ∈ {1,3}`, with 5-fold stratified CV, **refit on macro-F1**. It selected `n_estimators=400, max_depth=20, min_samples_leaf=1`.

**Q: Why max_depth=20? Doesn't that overfit? Should it be lower?**
The grid *chose* 20 empirically — and I can show the CV curve:

| max_depth | macro-F1 | balanced-acc | accuracy |
|---|---|---|---|
| 4 | 0.486 | 0.793 | 0.720 |
| 6 | 0.633 | 0.821 | 0.864 |
| 12 | 0.791 | 0.808 | 0.929 |
| **20** | **0.809** | 0.796 | 0.947 |
| None | 0.805 | 0.787 | 0.947 |

Three takeaways: (1) **shallow trees underfit badly** — macro-F1 collapses to 0.49 at depth 4. (2) Going from 20 to unlimited barely moves CV (0.809 → 0.805), so **depth isn't causing overfitting** — the RandomForest ensemble (400 bagged trees) absorbs deep individual trees; that's the *intended* way to use RF. (3) On this data each fold has ~4,200 rows, so a tree runs out of data around depth ~12 anyway — 20 is a soft cap that rarely binds. The clinching evidence: my **hold-out (0.816) is as good as CV (0.809)**, which is the opposite of overfitting. Fun nuance: balanced-accuracy actually peaks at depth 6 (catches more minorities but with terrible precision) — since macro-F1 balances both, 20 is the right pick.

**Q: What's your cross-validation strategy and why stratified?**
5-fold **StratifiedKFold** on the 90% training set. Stratified means each fold preserves the class proportions — essential when Other is <1% of the data; a plain random split could leave a fold with zero minority examples, making the score meaningless.

**Q: Why refit on macro-F1 instead of accuracy?**
Because accuracy rewards a model that ignores the minorities. Macro-F1 averages the per-class F1 equally, so the rare classes count as much as the dominant one. Optimizing the metric you actually care about is the whole point — tuning on accuracy would have selected a lazier model.

---

## F. Evaluation & metrics

**Q: What's your accuracy, and why isn't it the metric you'd report?**
Accuracy is 0.954 on the hold-out — but I lead with **macro-F1 (0.816)** and **balanced accuracy (0.790)**. Here's the intuition: 88% of rows are Category_1, so a model that blindly predicts Category_1 already "scores" 88% while being useless on everything else. Accuracy hides that. Macro-F1 and balanced accuracy weight the minority classes equally, so they reveal the model's *real* skill.

**Q: Explain your confusion matrix.**
On 590 hold-out rows: Category_1 is nearly perfect (510/522 correct). Category_2 gets 50/63 right, with 13 leaking to Category_1 — expected, since it's the closest big class. Other gets 3/5, with 2 misread as Category_1. The pattern is intuitive: errors flow *toward the majority class*, which is exactly what imbalance does, and class weighting keeps it from being worse.

**Q: Walk me through the classification report — which class is weakest?**
Per-class F1: Category_1 ≈ 0.97, Category_2 ≈ 0.81, Other ≈ 0.67. **Other is weakest**, which makes sense — it's only 5 hold-out rows drawn from classes that had 2–23 examples total. I present its number but I'm explicit that it's *indicative, not reliable* — you can't draw strong conclusions from 5 rows.

**Q: How do you know it generalizes and isn't overfitting?**
Two pieces of evidence. (1) The **CV and hold-out numbers agree** — macro-F1 0.809 (CV) vs 0.816 (hold-out); if it were overfitting, the hold-out would drop well below CV. (2) The **SHAP feature effects look the same on train and test** — the model relies on the same signal in both, not on memorized quirks.

**Q: What's ROC-AUC telling you that macro-F1 isn't?**
ROC-AUC (0.982, one-vs-rest macro) measures how well the model *ranks* classes across all thresholds — it's high because the predicted probabilities separate the classes well. Macro-F1 (0.816) measures the *hard-label decisions* at the default threshold. Both being reported is honest: the model ranks very well, and its default-threshold decisions are strong but leave a little on the table for rare classes (which threshold tuning could recover).

---

## G. Split strategy

**Q: How did you partition the data, and justify it.**
Stratified **90/10**: 10% is a held-out test set set aside up front and never touched during training (5,309 train / 590 test). Stratified so both parts keep the 88/11/1 class mix. The brief asked to set aside 10% as a test set — this is exactly that.

**Q: Where's the validation set — I only see train and test?**
The **5-fold cross-validation on the 90% is the validation.** Each of the five folds acts as a validation set once, so every training row gets validated on without ever touching the 10% test. CV is a more data-efficient form of validation than carving out a single fixed validation slice — valuable when the minority classes are small.

**Q: Why 10%?**
It's the brief's requirement, and it balances two needs: enough test rows to estimate performance (590), while leaving as much data as possible for training the rare classes. With classes this small, giving away much more than 10% would starve training.

---

## H. Tab 2 pipeline

**Q: How do you guarantee the exact same preprocessing as training?**
Because there *is* only one preprocessing. All cleaning + encoding + the model live inside a **single scikit-learn `Pipeline` object** that I `fit` once on the training data and save with joblib. Tab 2 loads that exact object and calls `.predict()`. The imputers' medians, the scaler's stats, and the encoder's category list were all learned at training time and are frozen inside it — there's no separate "app preprocessing" that could drift.

**Q: What if I upload a malformed file / wrong or extra columns?**
The app validates that the required Cols A–G are present and shows a clear error if any are missing; extra columns are ignored (I select the schema I need). Bad values are handled by the pipeline itself — Col3 non-numbers coerce to NaN then get imputed, unseen categories map to −1, and blank Col4 becomes `'missing'`. So it degrades gracefully instead of crashing, which is what the brief means by "handle edge cases."

**Q: Is this live or pre-computed?**
Fully live. Nothing about the uploaded file exists until you upload it — the predictions and confidences are computed on the spot by the loaded pipeline. I can prove it by uploading a fresh file and showing the results appear only after inference runs.

**Q: How would this scale to a much larger upload?**
Inference is a vectorized `predict` over the whole frame, so it scales roughly linearly and handles tens of thousands of rows comfortably. For very large files I'd stream/batch the read and paginate the table, and the SHAP explanation stays cheap because it's computed **only for the one row you click**, on demand.

---

## I. Agentic explainer

**Q: How does the "Explain" agent work end to end?**
You click a row → I compute **SHAP** for that single row using a TreeExplainer on the model, fold the contributions back to the 7 original columns, and build a compact, factual prompt listing each feature, its value, and its signed contribution. That prompt (plus a strict system message) goes to the LLM, which returns a plain-English "why this prediction." It only runs **on demand**, per the brief.

**Q: Can it hallucinate? How do you prevent it?**
I constrain it hard. The system prompt says *use only the SHAP contributions provided and do not invent features or numbers*, and the user prompt contains nothing but that row's real values and SHAP values. There's a **"Show the prompt" button** that displays exactly what's sent, so you can verify the model is grounded, not making things up. And `temperature=0` keeps it deterministic and factual.

**Q: Where do the feature contributions come from — what is SHAP doing?**
SHAP assigns each feature a signed number representing how much it pushed *this specific prediction* toward the predicted class, relative to a baseline — and those numbers add up to the model's output (additivity). For a RandomForest, TreeExplainer computes them exactly and fast. So "Col6 = +0.15" means Col6's value pushed this row toward its class by that much.

**Q: Why on-demand and not preloaded for all rows?**
The brief asks for it, and it's the right design: an LLM call per row across hundreds of rows would be slow and wasteful, and most rows never get inspected. Computing an explanation only when a human asks keeps it fast and cheap.

**Q: What if the LLM/API is down during the demo?**
There's a **deterministic SHAP fallback**: if the gateway is unreachable, the "Explain" button still returns a sentence built directly from the top SHAP contributions. So the demo never dead-ends on a network hiccup — which also demonstrates graceful degradation.

**Q: Are the SHAP plots pre-computed per class or generated live?**
Generated live, but computed *once and cached*. SHAP runs a single time over the whole hold-out producing a `(rows, features, classes)` array; the class selector just **re-slices** that array and redraws. So switching class is instant and always in sync with the loaded model — no static per-class images (only the two global-importance charts are saved PNGs).

---

## J. Dashboard & UX

**Q: Why Streamlit, and what's the architecture?**
Streamlit lets me deliver everything in one language (Python), so there's no JS/React plumbing to explain and the data-science code *is* the app. Architecture: `src/` holds the reusable pipeline, training, evaluation, SHAP, and agent modules; `app/streamlit_app.py` is a thin presentation layer that **loads frozen artifacts** (`model.joblib`, metrics, plots) and never retrains. Two tabs: story (Tab 1) and live pipeline + agent (Tab 2).

**Q: Show me an interactive element — how does it help the story?**
Tab 1 has three: an **any-column explorer** (pick Col1–Col7, see its cardinality/missingness/top-values/distribution live), a **per-class Col3 overlay** (does this feature separate the classes?), and a **live SHAP class selector**. They let the panel *interrogate* the data and model on demand rather than watch static slides — which is exactly the "interactive visualization" the brief rewards.

---

## K. Communication / business framing

**Q: One-sentence takeaway for a non-technical client?**
"The model correctly classifies about 95% of records and, more importantly, reliably separates the two business-relevant categories — and for any prediction it can tell you, in plain English, which fields drove it."

**Q: Where would you NOT trust this model?**
On the ultra-rare categories. `Other` is under 1% of the data and only 5 rows in the test set, so its numbers are indicative at best. I'd trust Category_1 and Category_2 confidently and treat any `Other` prediction as "flag for human review," not a firm answer.

**Q: What would you do with more time or more data?**
Target-encode the high-cardinality text columns to claw back the accuracy I traded for explainability; calibrate the probabilities; tune per-class decision thresholds; and — most impactful — **collect more rare-class examples**, since their weakness is a data-volume problem, not a modelling one.

**Q: If this went to production, what would you monitor?**
Input drift (are uploaded category distributions shifting, is the unseen-category `-1` rate rising?), prediction-mix drift, and per-class performance as labels arrive. A spike in unseen categories or a drop in Category_2 recall would be my early-warning signs to retrain.

---

## L. Curveballs

**Q: Your Other-class F1 is 0.67 on 5 rows — is that meaningful?**
Not on its own — I'd call it encouraging but statistically thin. Five rows can swing that F1 wildly, so I don't over-claim it; it's the honest limitation of classes that had 2–23 examples total.

**Q: You dropped Col2 — what if it secretly leaks the label?**
If it *did* correlate with the label it would be leakage of the worst kind (a near-unique ID standing in for the answer), so dropping it protects against exactly that. It's corrupted and near-unique, so it can't generalize regardless. If someone insisted, I'd test it by adding cheap format-features and checking CV — but I'd expect no honest lift.

**Q: Could there be data leakage anywhere in your pipeline?**
The main guard is that **all fitting happens inside the pipeline on the training folds only** — imputation medians, scaler stats, and encoder categories never see the test data. Tuning is done through the pipeline inside CV, so no preprocessing is fit on validation folds. The only global operation is deterministic label cleaning, which uses no target information.

**Q: Talk me through the one line you're least confident about.**
Honestly, the ordinal-vs-one-hot encoding trade — it costs a few macro-F1 points. I chose it deliberately for stable, column-level SHAP, but if the panel weighted raw accuracy over explainability, I'd revisit it with target encoding. (Naming a real trade-off, with the fix, reads as maturity — not weakness.)
