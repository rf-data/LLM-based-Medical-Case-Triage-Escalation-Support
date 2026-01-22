# imports 
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
