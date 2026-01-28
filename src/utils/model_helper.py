# imports
from openai import OpenAI
import json

from core.session import session


# -----------------
# LOAD LLM CLIENTS 
# -----------------

def get_openai_client() -> OpenAI:
    # if not hasattr(_thread_local, "client"):
    client = OpenAI()
    return client


# ---------------------------
# CONTENT CREATION FUNCTIONS
# ---------------------------

def content_creator_single(texts, scheme, allowed_values):
    CONTENT = f"""
Clinical report:
\"\"\"
{texts}
\"\"\"

You will receive multiple clinical reports.
For each report, return ONE JSON object with the fields:
{scheme}

{allowed_values}

Respond ONLY with a JSON array.
Do not include explanations.
"""
    return CONTENT 


def content_creator_batch(texts, scheme, allowed_values):
    reports = [
        {"id": i, "text": t}
        for i, t in enumerate(texts)
    ]

    CONTENT = f"""
Clinical report:
\"\"\"
{texts}
\"\"\"

You will receive multiple clinical reports.
For each report, return ONE JSON object with the fields:
{scheme}

{allowed_values}

Respond ONLY with a JSON array.
Do not include explanations.

Reports:
{json.dumps(reports, indent=2)}
"""
# - expected_action (boolean)
# - confidence (float)
# - uncertainty_level (string)
    return CONTENT 


# ----------------------
# ESCALATION FUNCTIONS  
# ----------------------

def single_escalation_by_llm(
    text: str,
    prompt: str,
    scheme: dict,
    allowed_values: str,
) -> dict:
    
    llm_model = session.llm_model
    if llm_model == "openai":
        client = get_openai_client()

    content = content_creator_single(text, scheme, allowed_values)

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": content},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
    )

    data = json.loads(response.choices[0].message.content)
    return normalize_llm_response(data)


def batch_escalation_by_llm(
    texts: list[str],
    prompt: str,
    scheme: dict,
    allowed_values: str,
) -> list[dict]:
    llm_model = session.llm_model
    if llm_model == "openai":
        client = get_openai_client()

    content = content_creator_batch(texts, scheme, allowed_values)

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": content},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
    )

    data = json.loads(response.choices[0].message.content)

    if not isinstance(data, list):
        raise TypeError(f"Batch LLM must return list[dict], got {type(data)}")

    return [normalize_llm_response(d) for d in data]

def normalize_llm_response(response: dict) -> dict:
    """
    Enforces a stable result schema for downstream code.
    """
    return {
        "expected_action": bool(
            response.get("expected_action", True)
        ),
        "severity": response.get("severity", "unknown"),
        "risk_factors": response.get("risk_factors", []),
        "uncertainty_level": str(response.get("uncertainty_level", "unknown")),
        "missing_information": response.get("missing_information", []), 
        "confidence": float(response.get("confidence", 0.0)),
    }
