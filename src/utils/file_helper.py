# import yaml
import json
import os
from pathlib import Path
import inspect
import numpy as np
from datetime import datetime
from functools import reduce
import pandas as pd
from typing import List
import io
import ast
import core.logger as log

# import hashlib

import utils.general_helper as gh
import utils.path_helper as ph


# -----------------
# DATAFRAME METHODS
# -----------------

def load_dfs(paths: List[str |Path] | str | Path, 
             index_col: str | None = None):
    if isinstance(paths, (Path, str)):
        paths = [paths]

    dfs_dict = {}
    for p in paths:
        df = pd.read_csv(p, index_col=index_col)
        dfs_dict[p] = df

    return dfs_dict


def df_preview(data: dict, logger=None):
    """
    Docstring for data_preview
    
    """

    for name, df in data.items():
        if logger:
            log.log_section(logger, f"EDA RAW DATA ({name})")
            logger.info("SHAPE:\t %s", df.shape) 
            logger.info("INFO:\n %s", info_as_string(df)) 
            logger.info("HEAD:\n %s %s", df.head(5), "\n") 
    
        else:
            print("\n")
            print("=" * 50 + "\n")
            print(f"--- EDA RAW DATA ({name}) --- {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ---\n")
            print("=" * 50 + "\n")
            print("SHAPE:\t", df.shape)
            print("INFO\n", info_as_string(df))
            print(f"HEAD:\n{df.head(5)}\n")

    return 


def col_name_correct(df_input, split_value):
    df = df_input.copy()

    col_to_rename = {}
    # name = ""
    for col in df.columns:
        parts = col.split(f"{split_value}")     # here: " "
        if len(parts) >= 2:
            col_to_rename[col] = parts[0]
            # name = parts[1]
        else:
            print(f"[INFO] No column name in df was splitted at '{split_value}'.")

    return df.rename(columns=col_to_rename)

def df_quick_check(df):
    print("[CHECK] SHAPE:\n", df.shape)
    print("\n[CHECK] HEAD:\n", df.head())
    return 

def parse_list_str(x):
    if not isinstance(x, str) or x.strip() == "":
        return []
    
    # Case 1: echte Python-Listenrepräsentation
    if x.startswith("[") and x.endswith("]"):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
        
    # Case 2: Fallback – kommasepariert
    if "," in x:
        return [v.strip() for v in x.split(",") if v.strip()]

    return []

# def parse_list_str(x):
#     if not isinstance(x, str) or x.strip() == "":
#         return []
#     try:
#         return ast.literal_eval(x)
#     except Exception:
#         return []
    
# def count_str_list_elements(x):
#     x_new = parse_list_str(x)

#     return len(x_new)

def info_as_string(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


def merge_dfs(df_list: List[pd.DataFrame], 
              on_cols: List[str] | str,
              suffix_col: str | None = None, 
              drop_cols: List[str] | str | None = None, 
              how: str = "inner") -> pd.DataFrame:
    
    if isinstance(on_cols, str):
        on_cols = [on_cols]

    if not on_cols:
        raise ValueError("on_cols must be specified for merge")
    
    dfs_clean = []
    # cols = []
    for i, df in enumerate(df_list):
        df = df.rename(
                columns={c: f"{c}_{i}"
                for c in df.columns
                if c == suffix_col}) 
        
        if drop_cols:
            df_clean = df.drop(columns=drop_cols, errors="ignore").copy() #.columns = [col for col in df.columns if col not in drop_cols]
        else: 
            df_clean = df.copy()

        for col in on_cols:
            if col not in df_clean.columns:
                raise ValueError(f"Column '{col}' is not in df '{i}'.")
            
        # cols.extend(df.columns)
        dfs_clean.append(df_clean)
         
    df_merged = reduce(
                    lambda left, right: pd.merge(
                        left,
                        right,
                        on=on_cols, 
                        how=how,
                    ),
                    dfs_clean
                )
    
    return df_merged




# ---------------------
# DICT / JSON METHODS
# ---------------------

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {
            make_json_safe(k): make_json_safe(v) 
            for k, v in obj.items()
            }
    if isinstance(obj, (list, tuple)):
        return [
            make_json_safe(v) 
            for v in obj
            ]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer): 
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if inspect.isfunction(obj):
        return gh.snapshot_single_function(obj)
        
    return obj


def save_dict(path: Path, data: dict) -> None:
    ph.ensure_dir(path) 
    data_new = make_json_safe(data)

    with path.open("w", encoding="utf-8") as f:
        try:
            json.dump(
                data_new,
                f,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            print(f"Dict saved as {ph.shorten_path(path, 3)}")
        except TypeError:
            print("NON-SERIALIZABLE:", type(data_new), repr(data_new))


def append_json(path: Path, data: dict) -> None:
    ph.ensure_dir(path) 
    data_new = make_json_safe(data)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data_new) + "\n")
        print(f"Appending data on {ph.shorten_path(path, 3)}")


def load_dict(path: Path) -> dict:
    ph.ensure_dir(path) 

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        print(f"Dict loaded: {ph.shorten_path(path, 3)}")
        return data


# -----------------
# TEXT METHODS
# -----------------

def save_text(path: Path, data: str):
    # make_text_safe(data)
    data_new = str(data)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(data_new)
    print(f"Text file saved as {ph.shorten_path(path, 3)}")



"""
logger.log_text(
    "red_flags.yaml",
    yaml.safe_dump(RED_FLAGS, sort_keys=True, allow_unicode=True)
)
"""
