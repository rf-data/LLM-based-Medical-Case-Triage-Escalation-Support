# imports
import json
from openai import OpenAI

from src.utils.prompt import prompt_v1, allowed_values_v1
from src.utils.json_scheme import scheme_v1


def llm_escalation(report_text: str) -> bool:
    # instanciate LLM
    client = OpenAI()

    # configurations
    SYSTEM_PROMPT = prompt_v1

    SCHEMA_INSTRUCTION = f"""
    Extract the following information and respond ONLY in valid JSON
    according to this schema:
    {scheme_v1}

    {allowed_values_v1}
    """

    # ??? 
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""
Clinical report:
\"\"\"
{report_text}
\"\"\"

{SCHEMA_INSTRUCTION}
"""
        },
    ]

    for attempt in range(2):        # retry once
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0,
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)

            escalation = bool(data["escalation_required"])
            uncertainty = str(data["uncertainty_level"]) 
            confidence = float(data["confidence"])

            if not isinstance(data["escalation_required"], 
                              bool):
                raise ValueError("Invalid 'escalation_required'")
            
            return data

        except Exception: 
            messages.append({
                "role": "system",
                "content": "Return VALID JSON ONLY. No text. No comments."
                    })

    return {
        "escalation_required": True,
        "confidence": 0.0,
        "uncertainty_level": "unknown"
            }

    #         all_llm(report_text)

    #     try:
    #         result = json.loads(response)
    #         return result["escalation_required"]
    #     except Exception:
    #         # Retry mit härterem Prompt
    #         continue

    # # Fallback: konservativ eskalieren
    # return True
