"""
On-demand LLM explainer for a single prediction (the Tab 2 "Explain" action).

Primary path : Azure OpenAI through the UHG API gateway (client-credentials token).
Fallback path: a deterministic sentence built straight from the SHAP contributions,
               so the Explain button always returns something useful even if the
               gateway is unreachable during the demo.

The agent only runs when the user clicks Explain on a specific row - never preloaded.
"""

import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv(override=True)  # read the .env file so the AZURE_* vars land in os.environ

# SYSTEM_PROMPT = the fixed instructions given to the model on every call; it pins the
# tone (concise, business-friendly) and forbids the model from inventing features/numbers
SYSTEM_PROMPT = (
    "You are a data science assistant explaining a single classification prediction to a "
    "business stakeholder. Use ONLY the SHAP feature contributions provided. Be concise "
    "(3-5 sentences), plain-spoken, and specific about which features pushed the prediction "
    "and in which direction. Do not invent features or numbers."
)

# _token_cache = a tiny in-memory store for the gateway token so we don't re-auth on every
# click. 'token' holds the bearer string, 'exp' is the epoch time it should be refreshed at.
_token_cache = {"token": None, "exp": 0.0}


def is_configured():
    """Return True only if the Azure gateway credentials exist in the environment.

    We check two env vars that come from the .env file (loaded above):
      - AZURE_CLIENT_ID       : the service-account id used for the token request
      - AZURE_OPENAI_ENDPOINT : the gateway URL we'd send the chat request to
    If either is missing we skip the LLM entirely and use the SHAP fallback, so the
    app still works on a machine that has no credentials.
    """
    return bool(os.environ.get("AZURE_CLIENT_ID") and os.environ.get("AZURE_OPENAI_ENDPOINT"))


def _get_token():
    """Fetch (or reuse) an OAuth access token for the gateway, caching it in memory.

    The gateway authenticates with a short-lived bearer token, so we request one with
    the client-credentials flow and cache it so every Explain click doesn't re-auth.
    """
    # if we still hold a token that hasn't hit its refresh time, reuse it as-is
    if _token_cache["token"] and time.time() < _token_cache["exp"]:
        return _token_cache["token"]

    # body = the client-credentials form fields the token endpoint expects
    body = {
        "grant_type": "client_credentials",              # the OAuth flow (no user, just an app identity)
        "scope": os.environ["AZURE_TOKEN_SCOPE"],         # which API the token is allowed to call
        "client_id": os.environ["AZURE_CLIENT_ID"],       # the service-account id
        "client_secret": os.environ["AZURE_CLIENT_SECRET"],  # the service-account password
    }
    # client = a short-lived HTTP client; POST the form and read the JSON reply
    with httpx.Client(timeout=120) as client:
        resp = client.post(
            os.environ["AZURE_TOKEN_URL"],  # the token endpoint URL
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=body,
        )
        resp.raise_for_status()  # turn any 4xx/5xx into an exception the caller can catch
        data = resp.json()       # data = parsed response, holds access_token + expires_in

    _token_cache["token"] = data["access_token"]  # the bearer token string we'll send as auth
    # store when it expires, minus a 120s safety margin so we refresh slightly early
    _token_cache["exp"] = time.time() + int(data.get("expires_in", 3000)) - 120
    return _token_cache["token"]


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


def _get_chat_client(temperature=0):
    """Build an AzureChatOpenAI handle pointed at the gateway and authed with our token.

    Shared by the row explainer and the dataframe Q&A agent so the auth/header setup
    lives in one place. temperature defaults to 0 (deterministic, factual).
    """
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],      # gateway base URL
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],      # Azure OpenAI REST version
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],  # the specific model deployment name
        temperature=temperature,
        azure_ad_token=_get_token(),                            # bearer token from the flow above
        default_headers={
            "projectId": os.environ.get("AZURE_PROJECT_ID", ""),  # gateway routing/billing header
            "model-usage-type": "prod",                            # gateway usage tag it requires
        },
    )


def fallback_explanation(explanation):
    """Build an explanation straight from SHAP, used when the LLM can't be reached.

    explanation : the explain_row() dict. We keep only the features that push toward
                  the predicted class and name the strongest few, so the Explain button
                  still returns something meaningful with zero LLM dependency.
    """
    # supports = up to the 3 strongest features whose SHAP pushes toward the predicted class
    supports = [f for f in explanation["top_features"] if f["direction"] == "supports"][:3]
    # parts = those features rendered as 'Col6 (value), Col1 (value), ...' for the sentence
    parts = ", ".join(f"{f['feature']} ({f['value']})" for f in supports) or "no single dominant feature"
    return (
        f"Predicted **{explanation['prediction']}** with {explanation['confidence']:.0%} confidence. "
        f"The strongest factors pushing toward this class were {parts}. "
        "(Generated directly from the SHAP contributions.)"
    )


def explain_prediction(explanation):
    """Return a plain-language explanation for one prediction (the public entry point).

    explanation : the explain_row() dict for the row the user clicked Explain on.
    Tries the LLM first; if credentials are missing or the call fails for any reason,
    it degrades gracefully to fallback_explanation() so the demo never dead-ends.
    """
    if not is_configured():  # no credentials -> skip straight to the deterministic summary
        return fallback_explanation(explanation)
    try:
        # client = the chat model handle, pointed at the gateway and authed with our token
        client = _get_chat_client()
        # resp = the model's reply; we send a system message (rules) + the user prompt
        system_prompt, user_prompt = build_messages(explanation)  # single source of truth
        resp = client.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return resp.content  # the generated explanation text
    except Exception as e:  # network/auth/gateway issues must not break the demo
        # e = whatever went wrong; show the SHAP fallback and note the error type
        return f"{fallback_explanation(explanation)}\n\n_(LLM unavailable: {type(e).__name__})_"


def answer_data_question(df, question):
    """Answer a natural-language question about a dataframe (the Tab 1 'chat with data').

    Uses a LangChain pandas DataFrame agent: the LLM is given the frame's schema, writes
    pandas code, executes it LOCALLY on `df`, and returns the result in plain English.

    Security note: the agent runs LLM-generated Python, so LangChain requires an explicit
    allow_dangerous_code flag. We only point it at the trusted local challenge data, never
    at an arbitrary user upload, which keeps the prompt-injection surface small.

    df       : the dataframe to answer questions about.
    question : the user's natural-language question.
    """
    if not is_configured():
        return "LLM not configured - set the AZURE_* variables in .env to enable data Q&A."
    try:
        from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

        agent = create_pandas_dataframe_agent(
            _get_chat_client(),
            df,
            agent_type="tool-calling",  # native function-calling (the reasoning model rejects ReAct's 'stop')
            allow_dangerous_code=True,  # required: the agent executes generated pandas code
            prefix=(
                "You are analysing a pandas dataframe named df. For every question, USE THE PYTHON "
                "TOOL to compute the answer on df, then reply in concise plain English that INCLUDES "
                "the computed numbers. Never return raw code as your final answer."
            ),
            verbose=False,
            max_iterations=8,
        )
        result = agent.invoke({"input": question})
        return result["output"] if isinstance(result, dict) else str(result)
    except Exception as e:
        return f"Couldn't answer that ({type(e).__name__}). Try rephrasing, e.g. 'how many rows are Category_2?'"
