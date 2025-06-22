import logging
import os 
from datetime import datetime




LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_PATH=os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(LOG_PATH, exist_ok=True)

LOG_FILE_PATH=os.path.join(LOG_PATH, LOG_FILE)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__ == "__main__":
    logging.info("Logging has been set up successfully.")
    logging.info("This is an info message.")
    logging.error("This is an error message.")
    logging.warning("This is a warning message.")
    logging.debug("This is a debug message.")
    logging.critical("This is a critical message.")


    