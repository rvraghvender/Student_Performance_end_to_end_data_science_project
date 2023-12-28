import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

############################################################
'''                       Additional things              '''

# Create a rotating file handler
""" 10 MB max size, keep 5 backup copies """ 
# rotating_handler = RotatingFileHandler(LOG_FILE_PATH, maxBytes=10*1024*1024, backupCount=5)  
# rotating_handler.setFormatter(logging.Formatter("[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"))
# logging.getLogger().addHandler(rotating_handler)


# Add console handler
"""
Adding a console handler allows you to see log messages in the console as well. 
This is particularly useful during development and debugging.
"""
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console_handler)

if __name__ == "__main__":
    logging.info("Loggin has started.")