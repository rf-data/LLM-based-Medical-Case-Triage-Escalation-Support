
scheme_v1 = {
          "expected_action": True,
          "severity": "low",            # | medium | high",
          "risk_factors": [],           # "string"
          "uncertainty_level": "low",   #  | medium | high",
          "missing_information": [],    # "string
          "confidence": 0.0
              }

scheme_v2 = {
        "severity": "low | medium | high",
        "uncertainty": "low | medium | high",
        "risk_factors": ["string"],
        "missing_information": ["string"],
        "confidence": 0.0
        }

def need_for_escalation(scheme: dict) -> bool:
    """
    Determine if escalation is needed based on the severity level in the scheme.

    Args:
        scheme (dict): The JSON scheme containing the severity level.
    """

