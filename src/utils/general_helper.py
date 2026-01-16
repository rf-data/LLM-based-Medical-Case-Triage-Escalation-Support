# imports
from dotenv import load_dotenv, find_dotenv
import subprocess
import json
import numpy as np
import os
from pathlib import Path
from typing import Union, Callable, Iterable
import inspect
import hashlib

from src.utils.session import session


def find_project_root() -> Path:
    start = Path(__file__).resolve()

    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise RuntimeError("Project root not found")


def ensure_dir(f_path: Union[str | Path]) -> Path:
    p = Path(f_path)
    
    target_dir = p.parent if p.suffix else p 
    target_dir.mkdir(parents=True, exist_ok=True)

    return p


def snapshot_single_function(fn: Callable) -> dict:
    src = inspect.getsource(fn)
    return {
        "name": fn.__name__,
        "qualname": fn.__qualname__,
        "module": fn.__module__,
        "source": src,
        "sha256": hashlib.sha256(src.encode("utf-8")).hexdigest(),
    }
# def describe_function(fn):
#     return {
#         "module": fn.__module__,
    
#         "name": fn.__name__,
#     }

def snapshot_dependent_functions(
    root_fn: Callable,
    dependencies: Iterable[Callable] | None = None,
) -> dict:
    snapshot = {
        "root": snapshot_single_function(root_fn),
        "dependencies": {},
    }

    if dependencies:
        for dep in dependencies:
            snapshot["dependencies"][dep.__name__] = snapshot_single_function(dep)

    return snapshot

# def hash_function_source(fn) -> str:
#     src = inspect.getsource(fn)
#     return src, hashlib.sha256(src.encode("utf-8")).hexdigest()


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
        return snapshot_single_function(obj)
        
    return obj


def shorten_path(path, n=3):
    p = Path(path).parts
    return "/".join(p[-n:])


def save_dict(path: Path, data: dict) -> None:
    ensure_dir(path) 
    data_new = make_json_safe(data)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data_new,
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        print(f"Dict saved as {shorten_path(path, 3)}")


def append_json(path: Path, data: dict) -> None:
    ensure_dir(path) 
    data_new = make_json_safe(data)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data_new) + "\n")
        print(f"Appending data on {shorten_path(path, 3)}")


def load_dict(path: Path) -> dict:
    ensure_dir(path) 

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        print(f"Dict loaded: {shorten_path(path, 3)}")
        return data


def iter_chunks(df, chunk_size=25):
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start : start + chunk_size]


def load_env_vars():
    """
    Load environment variables from .env files if available.
    """
    session_path = find_dotenv(filename=".env.session")
    if session_path and os.path.exists(session_path):
        load_dotenv(session_path, override=True)
        print("Variables from .env.session loaded")

    env_path = find_dotenv()
    if env_path and not session.env_loaded:
        load_dotenv(env_path)
        print("Variables from .env loaded")

    session.env_loaded = True
    session.save_session()



def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return "unknown"
