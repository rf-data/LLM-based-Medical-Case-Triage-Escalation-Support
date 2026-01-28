# imports
from dotenv import load_dotenv, find_dotenv
import subprocess

# import numpy as np
import os
from typing import Callable, Iterable
import inspect
import hashlib

from core.session import session

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
