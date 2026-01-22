# import yaml
import json
from pathlib import Path
import inspect
import numpy as np
# import hashlib

import src.utils.general_helper as gh
import src.utils.path_helper as ph


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if inspect.isfunction(obj):
        return gh.snapshot_single_function(obj)
        
    return obj


def save_dict(path: Path, data: dict) -> None:
    ph.ensure_dir(path) 
    data_new = make_json_safe(data)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data_new,
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        print(f"Dict saved as {ph.shorten_path(path, 3)}")


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


"""
logger.log_text(
    "red_flags.yaml",
    yaml.safe_dump(RED_FLAGS, sort_keys=True, allow_unicode=True)
)
"""
