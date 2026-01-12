## 
# imports
import hashlib
import os
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.escalation_llm import llm_escalation_single, llm_escalation_batch
import src.utils.general_helper as gh

# import threading
from openai import OpenAI

# _thread_local = threading.local()

def get_client() -> OpenAI:
    # if not hasattr(_thread_local, "client"):
    client = OpenAI()
    return client


def make_cache_key(report_text: str, 
                   prompt: str, 
                   namespace: str) -> str:
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8"))
    h.update(report_text.encode("utf-8"))
    h.update(namespace.encode("utf-8"))
    return h.hexdigest()


def load_from_cache(key: str, cache_dir: Path=None):
    if not cache_dir:
        cache_dir = Path(os.getenv("CACHE_DIR"))
    
    fn = cache_dir / f"{key}.json"
    gh.ensure_dir(fn)
    
    if fn.exists():
        with open(fn) as f:
            return json.load(f)
    return None

def save_to_cache(key: str, data: dict, cache_dir: Path=None):
    if not cache_dir:
        cache_dir = Path(os.getenv("CACHE_DIR"))

    fn = cache_dir / f"{key}.json"
    gh.ensure_dir(fn)
    
    with open(fn, "w") as f:
        json.dump(data, f, indent=2)
    

# def cached_escalation(report_text: str) -> dict:
#     key = make_cache_key(report_text, prompt)
        
#     cached = load_from_cache(key)
#     if cached is not None:
#             return cached

#     result = llm_escalation(report_text, 
#                             prompt=prompt,
#                             scheme=scheme, 
#                             allowed_values=allowed_values)
#     save_to_cache(key, result)
#     return result


# def cached_batch(texts: list[str]) -> list[dict]:
#     results = [None] * len(texts)
#     missing = []
#     missing_idx = []

#     # --- check cache ---
#     for i, text in enumerate(texts):
#         key = make_cache_key(text, prompt)
#         cached = load_from_cache(key)
#         if cached is not None:
#             results[i] = cached
#         else:
#             missing.append(text)
#             missing_idx.append(i)

#     # --- call LLM only for missing ---
#     if missing:
#         fresh = llm_escalation(
#                             missing, 
#                             prompt=prompt,
#                             scheme=scheme, 
#                             allowed_values=allowed_values)

#         for idx, res in zip(missing_idx, fresh):
#             key = make_cache_key(texts[idx], prompt)
#             save_to_cache(key, res)
#             results[idx] = res

#     return results

    # return cached_batch

def normalize_result(res: dict) -> dict:
    """
    Enforces a stable result schema for downstream code.
    """
    return {
        "expected_action": bool(
            res.get("expected_action")
            if "expected_action" in res
            else res.get("escalation_required", True)
        ),
        "confidence": float(res.get("confidence", 0.0)),
        "uncertainty_level": str(res.get("uncertainty_level", "unknown")),
    }

 
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
            key = make_cache_key(texts, 
                                 prompt, 
                                 namespace)
            cached = load_from_cache(key)
            if cached is not None:
                return cached

            result = llm_escalation_single(
                texts=texts,
                prompt=prompt,
                scheme=scheme,
                allowed_values=allowed_values,
            )
            save_to_cache(key, result)
            return result

        return cached_single
    
    if mode == "batch":

        def cached_batch(texts: list[str]) -> list[dict]:
            results = [None] * len(texts)
            missing_texts = []
            missing_idx = []

            for i, text in enumerate(texts):
                key = make_cache_key(text, 
                                 prompt, 
                                 namespace)
                cached = load_from_cache(key)
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
                    key = make_cache_key(texts[idx], 
                                        prompt, 
                                        namespace)
                    save_to_cache(key, res)
                    results[idx] = res

            return results
        
        return cached_batch
    
    else:
        raise ValueError(f"Unknown mode: {mode}")

def parallel_apply(texts, fn, max_workers=4):
    results = [None] * len(texts)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(fn, text): idx 
            for idx, text in enumerate(texts)
        }
        
        for future in as_completed(futures):
            idx = futures[future]

            try: 
                results[idx] = future.result()
            except Exception:
                 results[idx] = {
                    "expected_action": True,
                    "confidence": 0.0,
                    "uncertainty_level": "error"
                }
    
    return results


def batch_apply(
    texts: list[str],
    fn,                      # = fct_escalate (make_cached_escalation(prompt))
    batch_size: int = 5,
):
    """
    Apply an LLM escalation function in batches.
    Returns a list of results in the same order as texts.
    """

    results = [None] * len(texts)

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]

        # --- call batch function ---
        batch_results = fn(batch_texts)   # IMPORTANT: fn must accept list[str]

        if not isinstance(batch_results, list):
            raise TypeError(
                f"Batch function returned {type(batch_results)} instead of list[dict]. "
                f"Value: {batch_results}"
                        )

        if len(batch_results) != len(batch_texts):
            raise ValueError(
                f"Batch result length mismatch: "
                f"{len(batch_results)} != {len(batch_texts)}"
            )

        # --- assign back in correct order ---
        for i, res in enumerate(batch_results):
            if isinstance(res, dict):
                results[start + i] = res
            
            else:
                print(f"[ERROR] wrong dtype at index {i}")
                print(f"result:\n{res}\n")
                print(f"type:\t{type(res)}")
            
    return results

