## imports 
import logging
import sys
import os
from pathlib import Path 

import src.utils.general_helper as gh 


def has_file_handler(logger, log_path):
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            if Path(h.baseFilename) == Path(log_path):
                return True
    return False


def create_logger(name: str,
                  file_name: str, 
                  folder: str | Path | None = None,
                  level: str="info"
                  ) -> logging.Logger:

    """
    Create a configured logger with stdout + optional file logging.

    Parameters
    ----------
    name : str
        Logger name (e.g. "api", "health", "traffic")
    file_name : str | None
        Log file name (without .log). If None, no file logging.
    folder : str | Path | None
        Folder to save log files. If None, uses LOGS env var or "logs".
    level : str
        Log level ("debug", "info", "warning", "error", "critical")
    """

    level_dict = {
        "not_set": logging.NOTSET, 
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING, 
        "error": logging.ERROR, 
        # "exception": logging.exception, 
        "critical": logging.CRITICAL, 
    }

    log_level = level_dict.get(level.lower(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False  # VERY important with Uvicorn / Streamlit

    formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # --- stdout handler (always) ---
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # --- file handler (optional) ---
    if file_name:
        if not folder:
            gh.load_env_vars()
            folder = os.getenv("LOGS", "logs")
        
        log_path = gh.ensure_dir(folder)
        log_file = log_path / f"{file_name}.log"
        
        # check if file handler already exists
        if not has_file_handler(logger, log_file):
            file_handler = logging.FileHandler(
                                log_file, 
                                mode="a",
                                encoding="utf-8"
                                    )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
  
    return logger 


# if __name__ == "__main__":
#     create_logger()
