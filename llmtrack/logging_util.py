import os
import logging
from datetime import datetime
from typing import Optional

LOG_DIR = "logs"

def create_log_dir(name) -> str:
    subfolder = os.path.join(LOG_DIR, name)
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    return subfolder

def create_log_file(current_log_dir, log_file) -> str:
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(current_log_dir, f"{timestamp}.log")
    else:
        log_file = os.path.join(current_log_dir, log_file)
    return log_file

def setup_logger(name: str, log_file:Optional[str]=None, level: int = logging.INFO) -> logging.Logger:
    """Function setup as many loggers as you want"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.hasHandlers():
        # add a file handler
        current_log_dir = create_log_dir(name)
        log_file = create_log_file(current_log_dir, log_file)
        handler = logging.FileHandler(log_file)   
        formatter = logging.Formatter('%(asctime)s %(message)s')     
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
