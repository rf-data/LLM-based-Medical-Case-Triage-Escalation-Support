## imports 
import logging
import sys
import os
from pathlib import Path 

import src.utils.general_helper as gh 


def create_logger(name: str,
                  file_name: str, 
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

    # 🔒 Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # --- stdout handler (always) ---
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if file_name:
        gh.load_env_vars()
        log_dir = os.getenv("LOGS", "logs")
        log_path = gh.ensure_dir(log_dir)
    
    
        file_handler = logging.FileHandler(
                            log_path / f"{file_name}.log", 
                            mode="a",
                            encoding="utf-8"
                                )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger 


# if __name__ == "__main__":
#     create_logger()
