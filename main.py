from src.conponents.data_download import DataDownload
from src.conponents.data_tokenization import DataTokenization
from src.conponents.model_training import ModelTrainer

from src.logging import logger
import json


with open("config.json", "r") as f:
    config = json.load(f)

# STEP = "1. Data download"
# try:
#    logger.info(f">>>>>> stage {STEP} started <<<<<<") 
#    data_download = DataDownload(config)
#    data_download.main()
#    logger.info(f">>>>>> stage {STEP} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e

# STEP = "2. Data tokenization"
# try:
#    logger.info(f">>>>>> stage {STEP} started <<<<<<") 
#    data_tokenization = DataTokenization(config)
#    data_tokenization.main()
#    logger.info(f">>>>>> stage {STEP} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e

STEP = "3. Model training"
try:
   logger.info(f">>>>>> stage {STEP} started <<<<<<") 
   model_training = ModelTrainer(config)
   model_training.main()
   logger.info(f">>>>>> stage {STEP} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e