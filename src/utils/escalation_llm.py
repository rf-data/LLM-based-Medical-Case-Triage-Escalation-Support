# imports
import json
from openai import OpenAI

from src.core.mlflow_logger import get_experiment_logger
# from src.utils.escalation_baseline import baseline_escalation
from src.core.session import session
import src.utils.escalation_helper as esc
import src.utils.general_helper as gh


# _thread_local = threading.local()

def get_client() -> OpenAI:
    # if not hasattr(_thread_local, "client"):
    client = OpenAI()
    return client


def llm_escalation(df_input, save_df=True):
    # load logger
    logger = get_experiment_logger()
    
    df = df_input.copy()    
    # if mode == "llm":
    prompt = session.prompt
    scheme = session.json_scheme
    allowed_values = session.allowed_values
    namespace = session.namespace

    fct_escalate = make_cached_escalation(prompt=prompt,
                                        scheme=scheme, 
                                        allowed_values=allowed_values,
                                        namespace=namespace)

    mode = session.mode
    version_run = session.tags.get("vers_approach", "tba") # config["vers_run"]
    result_df = esc.df_iteration(df, fct_escalate)

    version = session.tags.get("vers_logic", "tba")
    fn_path = f"logic/{version}"
    func_name = session.dep_function_name
    func = session.dep_function

    snapshots = gh.snapshot_dependent_functions(fct_escalate,
                                                dependencies=[func])

    # log_text
    logger.log_text(f"{fn_path}/logic_complete.json",
                json.dumps(snapshots,
                        indent=2, ensure_ascii=False))

    logger.log_text(f"{fn_path}/logic_root.py",
                    snapshots["root"]["source"]) 

    code = snapshots["dependencies"][f"{func_name}"]["source"]
    logger.log_text(f"{fn_path}/logic_escalation.py",
                    code) 
    # set_tag

    logger.set_tag("source", f"{func_name} ({version})")
        
    dep_func_hashed = snapshots["dependencies"][f"{func_name}"]["sha256"]
    logger.set_tag("source_hashed", dep_func_hashed)
    
    UNCERTAINTY_TO_CONF = {
                        "low": 0.85,
                        "medium": 0.6,
                        "high": 0.3,
                    }
        
    df[f"pred_{mode}_{version_run}"] = result_df["expected_action"]
    df[f"confidence_llm_{version_run}"] = result_df["confidence"]
    df[f"uncertainty_llm_{version_run}"] = result_df["uncertainty_level"]
    df[f"confidence_derived_llm_{version_run}"] = (
                                        result_df["uncertainty_level"]
                                        .map(UNCERTAINTY_TO_CONF)
                                                    )
    # save df
    if save_df:
        esc.save_escalation_df(df, save_df)

    return df

 
def make_cached_escalation(*,
                           prompt: str, 
                           scheme: dict,
                           allowed_values: str,
                           namespace: str,
                           mode: str="batch"):
    
    """
    Factory that returns a cached escalation function.
    
    mode="single": fn(text: str) -> dict
    mode="batch":  fn(texts: list[str]) -> list[dict]
    """

    if mode == "single":
        def cached_single(texts: str) -> dict:
            key = esc.make_cache_key(texts, 
                                 prompt, 
                                 namespace)
            cached = esc.load_from_cache(key)
            if cached is not None:
                return cached

            result = llm_escalation_single(
                                    text=texts,
                                    prompt=prompt,
                                    scheme=scheme,
                                    allowed_values=allowed_values,
                                )
            
            esc.save_to_cache(key, result)
            
            return result

        return cached_single
    
    if mode == "batch":

        def cached_batch(texts: list[str]) -> list[dict]:
            results = [None] * len(texts)
            missing_texts = []
            missing_idx = []

            for i, text in enumerate(texts):
                key = esc.make_cache_key(text, 
                                 prompt, 
                                 namespace)
                cached = esc.load_from_cache(key)
                if cached is not None:
                    results[i] = cached
                else:
                    missing_texts.append(text)
                    missing_idx.append(i)

            if missing_texts:
                fresh = llm_escalation_batch(
                    texts=missing_texts,
                    prompt=prompt,
                    scheme=scheme,
                    allowed_values=allowed_values,
                )

                for idx, res in zip(missing_idx, fresh):
                    key = esc.make_cache_key(texts[idx], 
                                        prompt, 
                                        namespace)
                    esc.save_to_cache(key, res)
                    results[idx] = res

            return results
        
        return cached_batch
    
    else:
        raise ValueError(f"Unknown mode: {mode}")



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


def llm_escalation_single(
    text: str,
    prompt: str,
    scheme: dict,
    allowed_values: str,
) -> dict:
    client = get_client()

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
    return esc.normalize_result(data)


def llm_escalation_batch(
    texts: list[str],
    prompt: str,
    scheme: dict,
    allowed_values: str,
) -> list[dict]:
    client = get_client()

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

    return [esc.normalize_result(d) for d in data]



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
