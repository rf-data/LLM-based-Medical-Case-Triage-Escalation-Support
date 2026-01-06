# imports
from dotenv import load_dotenv, find_dotenv
import subprocess
from pathlib import Path
from typing import Union


def ensure_dir(f_path: Union[str | Path]) -> Path:
    p = Path(f_path)
    
    target_dir = p.parent if p.suffix else p 
    target_dir.mkdir(parents=True, exist_ok=True)

    return p


def iter_chunks(df, chunk_size=25):
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start : start + chunk_size]


def load_env_vars():
    """
    Load environment variables from an .env file if available.
    """

    # if session.env_loaded:
    #     return
    
    env_path = find_dotenv()
    if env_path:
        load_dotenv(env_path)
        print("Variables from .env loaded")

    # session_path = find_dotenv(filename=".env.session")
    # if session_path and os.path.exists(session_path):
    #     load_dotenv(session_path, override=True)
    #     print("Variables from .env.session loaded")
    
    # session.env_loaded = True



def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return "unknown"
