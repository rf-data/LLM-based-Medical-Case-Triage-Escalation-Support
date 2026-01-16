from pathlib import Path
import os
import src.utils.general_helper as gh

def clear_cache(cache_dir: Path=None) -> None:
    if not cache_dir or not cache_dir.exists():
        gh.load_env_vars()
        cache_dir = Path(os.getenv("CACHE_DIR"))
        cache_dir.mkdir(exist_ok=True)

    for f in cache_dir.iterdir():
        if f.is_file():
            f.unlink()

    print(f"[INFO] Cleared cache: {cache_dir}")

if __name__ == "__main__":
    clear_cache()
