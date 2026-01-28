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
from core.mlflow_logger import get_experiment_logger

# from concurrent.futures import ThreadPoolExecutor, as_completed
# from utils.escalation_baseline import baseline_escalation
from core.session import session
# from utils.escalation_llm import llm_escalation_single, llm_escalation_batch
import utils.general_helper as gh
import utils.file_helper as fh
import utils.path_helper as ph
from utils.model_helper import single_escalation_by_llm, batch_escalation_by_llm


def get_data_df(data):
    # load logger
    logger = get_experiment_logger()
    event_logger = logger.logger

    # load dataset path
    if data == "v1 - ambiguous":
        f_path = Path(os.getenv("AMBIGUOUS_DATA"))
        f_name = "data/data_generic/reports_ambiguous.csv" 

    elif data == "v1 - clear":
        f_path = Path(os.getenv("CLEAR_DATA"))
        f_name = "data/data_generic/reports_clear.csv"

    elif data == "v2":
        f_path = Path(os.getenv("DATA_V2"))
        f_name = "data/data_generic/escalation_dataset_v2.csv"

    # elif isinstance(data, Path):
    #     f_path = Path(data)

    # loading texts as df --> fct file load
    df = pd.read_csv(f_path, 
                     # index_col="Unnamed: 0"
                     )
    event_logger.info(f"df check -- head:\n{df.head(2)}\n")
    # print(f"")
    # logging
    logger.log_param("file name", f_name)
    logger.log_artifact(f_path)
    logger.log_param("name_dataset", data)
    logger.log_param("size_dataset", len(df)) 
    
    return df


def use_escalation_cache(*,
                           prompt: str, 
                           scheme: dict,
                           allowed_values: str,
                           namespace: str,
                           batch_mode: bool=True):
    
    """
    Factory that returns a cached escalation function.
    
    mode="single": fn(text: str) -> dict
    mode="batch":  fn(texts: list[str]) -> list[dict]
    """

    if batch_mode is False:
        def cached_single(texts: str) -> dict:
            key = make_cache_key(texts, 
                                 prompt, 
                                 namespace)
            cached = load_from_cache(key)
            if cached is not None:
                return cached

            result = single_escalation_by_llm(
                                    text=texts,
                                    prompt=prompt,
                                    scheme=scheme,
                                    allowed_values=allowed_values,
                                )
            
            save_to_cache(key, result)
            
            return result

        return cached_single
    
    elif batch_mode == True: 

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
                fresh = batch_escalation_by_llm(
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
        raise ValueError(f"Unknown mode: {batch_mode}")


def save_escalation_df(df): 
    # if save_df == True:
    now = session.now
    mode = session.mode
    version_run = session.tags.get("vers_approach", "tba") 

    folder = os.getenv("PATH_PROCESSED")
    ph.ensure_dir(folder)
    
    f_path = Path(f"{folder}/reports/{now}_{mode}_{version_run}_df.csv")
    ph.ensure_dir(f_path)
    df.to_csv(f_path)


def df_iteration(df, fct_escalate):
    # load logger
    logger = get_experiment_logger()
    event_logger = logger.logger

    now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    event_logger.info("Start processing reports: %s", now)
    session.now = now
    mode = session.mode
    # session.save_session()

    start = time.perf_counter()
    all_results = []
    
    if mode == "llm":
        for i, chunk in enumerate(gh.iter_chunks(df, chunk_size=25), 
                                  start=1):
            event_logger.info("Processing chunk %s", i)

            event_logger.info("'Chunk' columns:\n%s", 
                              chunk.columns.tolist())

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

    elif mode == "rule":
        texts = df["report_text"].tolist()
        for i, txt in enumerate(texts):
            result = fct_escalate(txt)
            all_results.append(result)

    elapsed = time.perf_counter() - start
    
    # creating + cleaning df
    result_df = pd.DataFrame(all_results)

    if mode == "llm": 
        UNCERTAINTY_TO_CONF = {
                            "low": 0.85,
                            "medium": 0.6,
                            "high": 0.3,
                        }
        
        result_df["confidence_derived"] = (result_df["uncertainty_level"])\
                                            .map(UNCERTAINTY_TO_CONF) 
        result_df["risk_factors"] = result_df["risk_factors"].apply(fh.parse_list_str)
        result_df["missing_information"] = result_df["missing_information"].apply(fh.parse_list_str)
        # result_df_corr = normalize_escalation_df(result_df)

    # logging
    event_logger.info("\n[CHECK] 'result_df' columns:\n %s\n", result_df.columns.tolist())

    logger.log_metric("runtime_total_sec", 
                      elapsed)
    logger.log_metric("runtime_per_sample_sec", 
                      (elapsed / len(df)))

    session.run_time = [elapsed, (elapsed / len(df))]
    session.save_snapshot()
    
    return result_df

# def normalize_escalation_df(df: pd.DataFrame) -> pd.DataFrame:
#     # --- numeric ---
#     numeric_cols = [
#         "confidence",
#         "confidence_derived",
#     ]

#     for col in numeric_cols:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors="coerce")

#     # --- categorical / strings ---
#     str_cols = [
#         "severity",
#         "uncertainty_level",
#         "expected_action",
#     ]

#     for col in str_cols:
#         if col in df.columns:
#             df[col] = df[col].astype(str).str.strip().str.lower()

#     # --- boolean targets ---
#     bool_cols = [
#         "expected_action_llm",
#         "expected_action_final",
#     ]

#     for col in bool_cols:
#         if col in df.columns:
#             df[col] = df[col].astype("boolean")

#     return df

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
        folder = os.getenv("CACHE_DIR")
        cache_dir = Path(f"{folder}")
    
    fn = cache_dir / f"{key}.json"
    ph.ensure_dir(fn)
    
    if fn.exists():
        with open(fn) as f:
            return json.load(f)
    return None

def save_to_cache(key: str, data: dict, cache_dir: Path=None):
    if not cache_dir:
        cache_dir = Path(os.getenv("CACHE_DIR"))

    fn = cache_dir / f"{key}.json"
    ph.ensure_dir(fn)
    
    with open(fn, "w") as f:
        json.dump(data, f, indent=2)


def batch_apply(
    texts: list[str],
    fn,                      # = fct_escalate (make_cached_escalation(prompt))
    batch_size: int = 5,
):
    """
    Apply an LLM escalation function in batches.
    Returns a list of results in the same order as texts.
    """
    # setup logger
    logger = get_experiment_logger()
    event_logger = logger.logger

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
                event_logger.error(f"wrong dtype at index {i}"
                                   f"result:\n{res}\n"
                                   f"type:\t{type(res)}")
            
    return results



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
