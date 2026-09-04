# EY Data Science Take-Home Challenge

A supervised classification solution delivered as an interactive **Streamlit** dashboard.

- **Tab 1 — Exploration, Preprocessing & Model Results:** the end-to-end data-science story
  (EDA, cleaning, feature engineering, model training/tuning, evaluation, split strategy).
- **Tab 2 — Hold-Out Prediction Pipeline:** upload a hold-out file (Columns A–G); the exact
  training pipeline runs live to produce predictions, and each row has an **AI "Explain"**
  action that grounds its answer in that row's SHAP contributions.

## Project structure
```
src/preprocessing.py   # the sklearn pipeline: cleaning, encoding, model, grid search
src/train.py           # clean -> stratified split -> tune -> save model + preprocessor + metrics
src/evaluate.py        # load artifacts -> score hold-out -> per-row results + SHAP
src/explain.py         # SHAP helpers (TreeExplainer, explain_row, importances)
src/plots.py           # RF / SHAP importance + SHAP summary/scatter (train & test)
src/agent.py           # on-demand LLM explainer (Groq open-source model) + SHAP fallback
app/streamlit_app.py   # the two-tab dashboard
eda.ipynb              # exploration notebook
models/                # saved model, preprocessor, metrics, plots (committed)
PRESENTATION_PREP.md   # interview / presentation notes
```

## How to run

### 1. Prerequisites
- Python **3.12**
- git

### 2. Clone
```bash
git clone https://github.com/SubhamIO/EY_takehome_assignment.git
cd EY_takehome_assignment
```

### 3. Create & activate a virtual environment
**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
**Windows (PowerShell)**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
> If `requirements.txt` doesn't yet include it, also run `pip install langchain-groq` for the LLM agent.

### 5. (Optional) Configure the LLM agent
The `.env` file is **not** committed (it holds a secret). Create it in the project root only if you
want the live LLM explanations; otherwise the "Explain" button falls back to a deterministic
SHAP-based summary. The agent uses [Groq](https://console.groq.com/) to run an open-source model
(`openai/gpt-oss-120b` by default) — get a free API key from the Groq console.
```
GROQ_API_KEY=<your groq api key>
GROQ_MODEL=openai/gpt-oss-120b
```
`GROQ_MODEL` is optional; omit it to use the default above.

### 6. Run the app
The trained `models/` and the CSVs are already in the repo, so you can launch immediately:
```bash
streamlit run app/streamlit_app.py
```
Opens at http://localhost:8501 → go to **Tab 2**, upload `holdout.csv`, click a row → **Explain**.

### 7. (Optional) Rebuild all artifacts from scratch
```bash
python src/train.py      # -> models/model.joblib, preprocessor.joblib, metrics.json, train.csv, holdout.csv
python src/evaluate.py   # -> confusion_matrix.csv, holdout_metrics.json, holdout_predictions.csv
python src/plots.py      # -> models/plots/*.png
```

## Notes
- The saved `.joblib` artifacts were produced with the versions pinned in `requirements.txt`
  (scikit-learn 1.9.0). Install from that file for guaranteed compatibility; if a load error ever
  occurs, just re-run the step 7 scripts to regenerate them locally.
- `holdout.csv` contains Columns A–G only (no label) — it's the file to upload in Tab 2.