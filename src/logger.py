import os
import yaml
import logging
from pathlib import Path 
from src.utils import get_config_data

config = get_config_data()

logs_dir = config["logs"].get("log_dir", ".")
default_file_name = "app.log"  # Define a default filename
file_name = config["logs"].get("file_name", default_file_name)
log_dir_path = Path(__file__).parent.parent / logs_dir
log_file_path = Path(log_dir_path) / file_name

log_level = config["logs"].get("log_level", logging.INFO)

# Ensure the logs directory exists
log_dir_path.mkdir(parents=True, exist_ok=True)  # Create logs dir if needed

# Create a basic FileHandler
log_handler = logging.FileHandler(log_file_path)
log_handler.setFormatter(logging.Formatter("[%(asctime)s] %(lineno)d %(name)s %(levelname)s %(message)s"))

# Get the root logger
logger = logging.getLogger('app_logger')
logger.setLevel(logging.INFO) 
logger.addHandler(log_handler)