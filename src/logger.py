import logging
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

LOG_DIR = os.path.join(script_dir, 'logs')
os.makedirs(LOG_DIR, exist_ok=True) 


LOG_FILE = 'app.log'

# Construct the full log file path
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Create a basic FileHandler
log_handler = logging.FileHandler(LOG_FILE_PATH)
log_handler.setFormatter(logging.Formatter("[%(asctime)s] %(lineno)d %(name)s %(levelname)s %(message)s"))

# Get the root logger
logger = logging.getLogger('app_logger')
logger.setLevel(logging.INFO) 
logger.addHandler(log_handler)


if __name__ == '__main__':
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')