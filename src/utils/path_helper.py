# imports 
import os
from pathlib import Path
from typing import Union

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


def shorten_path(path, n=3):
    p = Path(path).parts
    return "/".join(p[-n:])


def create_save_path(folder_name, name_suffix, file_suffix):
    # as lazy imports
    import utils.general_helper as gh
    from core.session import session
    
    gh.load_env_vars()
    folder = os.getenv("PATH_EVALUATED", None)
    now = session.now
    mode = session.mode
    version_run = session.tags.get("vers_approach", "tba") 

    f_path = Path(f"{folder}/{folder_name}/{now}_{mode}_{version_run}_{name_suffix}{file_suffix}")
    ensure_dir(f_path)
    
    return f_path