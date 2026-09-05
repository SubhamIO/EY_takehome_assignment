"""
On-demand LLM explainer for a single prediction (the Tab 2 "Explain" action).

Primary path : Groq (open-source hosted models) via GROQ_API_KEY.
Fallback path: a deterministic sentence built straight from the SHAP contributions,
               so the Explain button always returns something useful even if the
               Groq API is unreachable during the demo.

explain_prediction() returns {"text", "source", "is_llm"} so the caller can render a
clear badge showing whether the answer came from the LLM or the SHAP fallback.

The agent only runs when the user clicks Explain on a specific row - never preloaded.
"""

import os

from dotenv import load_dotenv

load_dotenv(override=True)  # read the .env file so GROQ_API_KEY lands in os.environ

# SYSTEM_PROMPT = the fixed instructions given to the model on every call; it pins the
# tone (business-friendly, moderately detailed) and forbids the model from inventing
# features/numbers
SYSTEM_PROMPT = (
    "You are a data science assistant explaining a single classification prediction to a "
    "business stakeholder. Use ONLY the SHAP feature contributions provided - never invent "
    "features or numbers. Write 5-8 sentences in plain language. Structure the explanation as: "
    "(1) state the predicted class and confidence, (2) walk through the features that pushed "
    "TOWARD the predicted class, naming each one, its value, and describing its SHAP contribution "
    "as a positive push, (3) walk through any features that pushed AGAINST the predicted class "
    "(i.e. toward a different class), describing their SHAP contribution as a negative pull, and "
    "(4) close with one sentence on which single feature was the most influential overall and why. "
    "Be specific about direction (positive/negative, supports/opposes) for every feature you mention."
)

# GROQ_MODEL = which hosted open-source model to call; overridable via env without a code change
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


def is_configured():
    """Return True only if a Groq API key exists in the environment.

    We check GROQ_API_KEY (from the .env file loaded above). If it's missing we skip
    the LLM entirely and use the SHAP fallback, so the app still works on a machine
    that has no credentials.
    """
    return bool(os.environ.get("GROQ_API_KEY"))


def _format_features(explanation):
    """Turn the SHAP top-features list into readable bullet lines for the prompt.

    explanation : the dict returned by explain_row() - we only use its 'top_features',
                  a list of {feature, value, shap, direction} entries.
    Returns one text line per feature, e.g. '- Col6 = "..." | SHAP +0.145 (supports ...)'.
    """
    rows = []  # rows = the formatted text lines we build up, one per feature
    for f in explanation["top_features"]:  # f = a single feature's contribution dict
        rows.append(
            f"- {f['feature']} = {f['value']!r} | SHAP {f['shap']:+.3f} ({f['direction']} the prediction)"
        )
    return "\n".join(rows)  # join the lines into one block of text


def _build_prompt(explanation):
    """Assemble the user-message text sent to the LLM for one prediction.

    explanation : the explain_row() dict, with 'prediction', 'confidence' and
                  'top_features'. We fold those into a compact, factual prompt so the
                  model explains the numbers we give it rather than inventing any.
    """
    return (
        f"The model predicted '{explanation['prediction']}' with confidence "
        f"{explanation['confidence']:.0%}.\n\n"
        "Feature contributions for THIS row (SHAP, signed toward the predicted class):\n"
        f"{_format_features(explanation)}\n\n"
        "Explain in plain language why the model made this prediction and which features mattered most."
    )


def build_messages(explanation):
    """Return the exact (system_prompt, user_prompt) pair the LLM call will use.

    Exposed so the dashboard can display the dynamically built prompt. Because both this
    and explain_prediction() read from here, what we show the user is guaranteed to be
    what actually gets sent to the model.

    explanation : the explain_row() dict for the selected row.
    """
    return SYSTEM_PROMPT, _build_prompt(explanation)


def _feature_sentence(f):
    """Render one feature's contribution as a short, direction-aware clause.

    f : a single {feature, value, shap, direction} entry from explain_row()'s
        'top_features'. Phrases the SHAP sign in plain words rather than just the number.
    """
    strength = "strongly" if abs(f["shap"]) >= 0.10 else "moderately" if abs(f["shap"]) >= 0.03 else "slightly"
    verb = "pushed toward" if f["direction"] == "supports" else "pulled away from"
    return f"**{f['feature']}** = {f['value']!r} {strength} {verb} this class (SHAP {f['shap']:+.3f})"


def fallback_explanation(explanation):
    """Build an explanation straight from SHAP, used when the LLM can't be reached.

    explanation : the explain_row() dict. We separate the features into those that push
                  toward the predicted class and those that pull away from it, and narrate
                  each group by strength, so the Explain button still returns a fairly
                  complete, direction-aware summary with zero LLM dependency.
    """
    top_features = explanation["top_features"]
    # sort by absolute SHAP so the most influential features are named first in each group
    ranked = sorted(top_features, key=lambda f: abs(f["shap"]), reverse=True)
    supports = [f for f in ranked if f["direction"] == "supports"]
    opposes = [f for f in ranked if f["direction"] != "supports"]

    lines = [
        f"Predicted **{explanation['prediction']}** with {explanation['confidence']:.0%} confidence."
    ]

    if supports:
        support_clauses = "; ".join(_feature_sentence(f) for f in supports[:4])
        lines.append(f"Pushing toward this outcome: {support_clauses}.")
    else:
        lines.append("No feature pushed strongly toward this outcome on its own.")

    if opposes:
        oppose_clauses = "; ".join(_feature_sentence(f) for f in opposes[:4])
        lines.append(f"Pulling away from this outcome (toward another class): {oppose_clauses}.")

    top = ranked[0] if ranked else None
    if top:
        direction_word = "supporting" if top["direction"] == "supports" else "opposing"
        lines.append(
            f"Overall, **{top['feature']}** was the single most influential feature "
            f"(SHAP {top['shap']:+.3f}, {direction_word} the predicted class)."
        )

    return " ".join(lines)


def _tag_source(text, source, is_llm):
    """Prefix an explanation with a one-line badge naming where it came from.

    text    : the explanation body (from the LLM or the SHAP fallback).
    source  : a short human-readable label, e.g. 'LLM (Groq - openai/gpt-oss-120b)'
              or 'SHAP fallback (no LLM configured)'.
    is_llm  : True if this text came from the LLM, False for the deterministic fallback.
    Every explanation - success or fallback - passes through here and is returned as a
    dict, so the dashboard can render an unambiguous badge (not just bolded text buried
    in the paragraph) showing which path produced the answer on screen.
    """
    return {"text": text, "source": source, "is_llm": is_llm}


def explain_prediction(explanation):
    """Return a plain-language explanation for one prediction (the public entry point).

    explanation : the explain_row() dict for the row the user clicked Explain on.
    Tries the LLM first; if credentials are missing or the call fails for any reason,
    it degrades gracefully to fallback_explanation() so the demo never dead-ends.

    Returns a dict: {"text": <explanation string>, "source": <human-readable label>,
    "is_llm": <bool>} so the UI can show a clear badge for which path produced it,
    rather than relying on wording inside the explanation text.
    """
    if not is_configured():  # no credentials -> skip straight to the deterministic summary
        return _tag_source(
            fallback_explanation(explanation),
            "SHAP fallback (no GROQ_API_KEY configured)",
            is_llm=False,
        )
    try:
        from langchain_groq import ChatGroq

        # llm = the chat model handle, pointed at Groq's hosted open-source model
        llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model=GROQ_MODEL,
            temperature=0,  # deterministic output for a factual task
        )
        # resp = the model's reply; we send a system message (rules) + the user prompt
        system_prompt, user_prompt = build_messages(explanation)  # single source of truth
        resp = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return _tag_source(resp.content, f"LLM (Groq - {GROQ_MODEL})", is_llm=True)
    except Exception as e:  # network/auth/gateway issues must not break the demo
        # e = whatever went wrong; show the SHAP fallback and note the error type
        fallback_text = f"{fallback_explanation(explanation)}\n\n_(LLM unavailable: {type(e).__name__})_"
        return _tag_source(
            fallback_text,
            f"SHAP fallback (LLM call failed: {type(e).__name__})",
            is_llm=False,
        )