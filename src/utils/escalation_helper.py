## 
# imports
import hashlib
import os
from pathlib import Path
from datetime import datetime
import time
# import mlflow
import pandas as pd
import json
from src.core.mlflow_logger import get_experiment_logger

# from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.escalation_baseline import baseline_escalation
from src.core.session import session
from src.utils.escalation_llm import llm_escalation_single, llm_escalation_batch
import src.utils.general_helper as gh

# import threading
from openai import OpenAI

# _thread_local = threading.local()

def get_client() -> OpenAI:
    # if not hasattr(_thread_local, "client"):
    client = OpenAI()
    return client

def get_data_df(data):
    # load logger
    logger = get_experiment_logger()

    # load dataset path
    if data == "ambiguous":
        f_path = Path(os.getenv("AMBIGUOUS_DATA"))
        f_name = "data/data_generic/reports_ambiguous.csv" 

    elif data == "clear":
        f_path = Path(os.getenv("CLEAR_DATA"))
        f_name = "data/data_generic/reports_clear.csv"

    elif data == "version_2":
        f_path = Path(os.getenv("DATA_V2"))
        f_name = "data/data_generic/escalation_dataset_v2.csv"

    elif isinstance(data, Path):
        f_path = Path(data)

    # loading texts as df --> fct file load
    df = pd.read_csv(f_path, 
                     # index_col="Unnamed: 0"
                     )
    print(f"df check -- head:\n{df.head(2)}\n")
    # print(f"")
    # logging
    logger.log_artifact(f_name)
    logger.log_artifact(f_path)
    logger.log_param("name_dataset", data)
    logger.log_param("size_dataset", len(df)) 
    
    return df


def case_escalation(df_input, save_df=True):
    df = df_input.copy()

    mode = session.mode

    if mode == "baseline":  # LLM
       fct_escalate = baseline_escalation
    
    elif mode == "llm":
        prompt = session.prompt
        scheme = session.json_scheme
        allowed_values = session.allowed_values
        namespace = session.namespace

        fct_escalate = make_cached_escalation(prompt=prompt,
                                                 scheme=scheme, 
                                                 allowed_values=allowed_values,
                                                 namespace=namespace)

    else:
        print(f"Unknown mode: {mode}")
        return 

    session.logic_function = fct_escalate

    version_run = session.tag.vers_approach # config["vers_run"]
    result_df = df_iteration(df, fct_escalate)

    if mode == "llm": 
        func_name = session.dep_function_name
        func = session.dep_function

        snapshots = gh.snapshot_dependent_functions(fct_escalate,
                                                dependencies=[func])

        version = session.tag.vers_logic
        fn_path = f"logic/{version}"

        # log_text
        logger.log_text(f"{fn_path}/logic_complete.json",
                json.dumps(snapshots,
                        indent=2, ensure_ascii=False))

        logger.log_text(f"{fn_path}/logic_root.py",
                    snapshots["root"]["source"]) 

        code = complete["dependencies"][f"{func_name}"]["source"]
        logger.log_text(f"{fn_path}/logic_escalation.py",
                    code) 
        # set_tag

        logger.set_tag("source", f"{func_name} ({version})")
        
        dep_func_hashed = complete["dependencies"][f"{func_name}"]["sha256"]
        logger.set_tag("source_hashed", dep_func_hashed)
        # dep_func_name = 
        

        # log_dict["logic"] = [snapshots, 
        #                      ("root", snapshots["root"]), 
        #                      (f"{func_name}", snapshots["dependencies"][f"{func_name}"])]

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

    else:
        df[f"pred_{mode}_{version_run}"] = result_df

        snapshot_dict = gh.snapshot_single_function(fct_escalate)
        log_dict["logic"] = snapshot_dict

    # save df
    if save_df == True:
        folder = Path(os.getenv("PROCESSED"))
        gh.ensure_dir(folder)
        f_path = folder / f"{now}_reports_{mode}_{version_run}.csv"
        df.to_csv(f_path)

    if isinstance(save_df, Path):
        f_path = Path(save_df)
        gh.ensure_dir(f_path)

        df.to_csv(f_path)

    return df, log_dict


def df_iteration(df, fct_escalate, log_dict, mode):
    now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    print(f"[INFO] Start processing reports: {now}")

    start = time.perf_counter()
    all_results = []
    
    for i, chunk in enumerate(gh.iter_chunks(df, chunk_size=25), start=1):
        print(f"[INFO] Processing chunk {i}")

        texts = chunk["report_text"].tolist()

        chunk_results = batch_apply(
                                texts=texts,
                                fn=fct_escalate,
                                batch_size=5
                                    ) 
        
        assert isinstance(chunk_results, list)
        assert all(isinstance(r, dict) for r in chunk_results)
        # if isinstance(chunk_results, dict):
        #     chunk_results = [chunk_results]
            
        all_results.extend(chunk_results)

    elapsed = time.perf_counter() - start
    
    result_df = pd.DataFrame(all_results)
    log_dict["runtime_total_sec"] = elapsed
    log_dict["runtime_per_sample_sec"] = (elapsed / len(df))
    log_dict["run_name"] = f"{now}_{mode}"

    return result_df, log_dict, now

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

# def parallel_apply(texts, fn, max_workers=4):
#     results = [None] * len(texts)

#     with ThreadPoolExecutor(max_workers=max_workers) as pool:
#         futures = {
#             pool.submit(fn, text): idx 
#             for idx, text in enumerate(texts)
#         }
        
#         for future in as_completed(futures):
#             idx = futures[future]

#             try: 
#                 results[idx] = future.result()
#             except Exception:
#                  results[idx] = {
#                     "expected_action": True,
#                     "confidence": 0.0,
#                     "uncertainty_level": "error"
#                 }
    
#     return results


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

