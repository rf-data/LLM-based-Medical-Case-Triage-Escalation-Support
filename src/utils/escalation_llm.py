# imports
import json
# from openai import OpenAI

import src.utils.escalation_helper as eh

# def llm_escalation(texts: str | list[str], 
#                    prompt: str, 
#                    scheme: dict,
#                    allowed_values: str) -> dict:
#     # instanciate LLM
#     client = eh.get_client()

#     if isinstance(texts, list):
#         CONTENT = content_creator_batch(texts, scheme, allowed_values)

#     elif isinstance(texts, str):
#         CONTENT = content_creator_single(texts, scheme, allowed_values)

#     else:
#         raise TypeError("Unknown data type")
    
#     messages = [
#         {"role": "system", "content": prompt},
#         {
#             "role": "user",
#             "content": CONTENT
#         }
#                 ]

#     for attempt in range(2):        # retry once
#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=messages,
#                 temperature=0.0,
#             )
            
#             content = response.choices[0].message.content
#             data = json.loads(content)

#             # escalation = bool(data["expected_action"])
#             # uncertainty = str(data["uncertainty_level"]) 
#             # confidence = float(data["confidence"])

#             if not isinstance(data["expected_action"], 
#                               bool):
#                 raise ValueError("Invalid 'expected_action'")
            
#             return data

#         except Exception: 
#             messages.append({
#                 "role": "system",
#                 "content": "Return VALID JSON ONLY. No text. No comments."
#                     })

#     return {
#         "expected_action": True,
#         "confidence": 0.0,
#         "uncertainty_level": "unknown"
#             }


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


        # data = llm_escalation_batch(texts)


#     if isinstance(texts, str):
#         CONTENT = f"""
# Clinical report:
# \"\"\"
# {texts}
# \"\"\"

# Extract the following information and respond ONLY in valid JSON
# according to this schema:
# {scheme}

# {allowed_values}

# Respond ONLY with a JSON array.
# Do not include explanations.

# Reports:
# {json.dumps(texts, indent=2)}
#             """

    # ??? 
    
# def llm_escalation_batch(texts: list[str], prompt: str, scheme)
    #         all_llm(report_text)

    #     try:
    #         result = json.loads(response)
    #         return result["escalation_required"]
    #     except Exception:
    #         # Retry mit härterem Prompt
    #         continue

    # # Fallback: konservativ eskalieren
    # return True

def llm_escalation_single(
    text: str,
    prompt: str,
    scheme: dict,
    allowed_values: str,
) -> dict:
    client = eh.get_client()

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
    return eh.normalize_result(data)


def llm_escalation_batch(
    texts: list[str],
    prompt: str,
    scheme: dict,
    allowed_values: str,
) -> list[dict]:
    client = eh.get_client()

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

    return [eh.normalize_result(d) for d in data]
