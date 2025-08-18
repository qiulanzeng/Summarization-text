from src.pipeline.step1_data_download import DataDownloadPipeline
from src.pipeline.step2_data_tokenization import DataTokenizationPipeline
from src.pipeline.step3_model_training import ModelTrainingPipeline
# from src.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
# from src.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
from src.logging import logger


# STEP = "1. Data download"
# try:
#    logger.info(f">>>>>> stage {STEP} started <<<<<<") 
#    data_download = DataDownloadPipeline()
#    data_download.main()
#    logger.info(f">>>>>> stage {STEP} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e

# STEP = "2. Data tokenization"
# try:
#    logger.info(f">>>>>> stage {STEP} started <<<<<<") 
#    data_tokenization = DataTokenizationPipeline()
#    data_tokenization.main()
#    logger.info(f">>>>>> stage {STEP} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e

STEP = "3. Model training"
try:
   logger.info(f">>>>>> stage {STEP} started <<<<<<") 
   model_training = ModelTrainingPipeline()
   model_training.main()
   logger.info(f">>>>>> stage {STEP} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e