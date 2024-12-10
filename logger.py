import logging
import os
from config import Config

os.makedirs('logs', exist_ok=True)
logger = logging.getLogger('RL_GridWorld')
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(Config.LOG_FILE)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

logger.info("Logger initialized and ready.")
