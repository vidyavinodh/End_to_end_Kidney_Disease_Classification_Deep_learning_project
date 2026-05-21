from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline
import os

# MLflow / DagsHub credentials
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/vidyavinodh1982/End_to_end_Kidney_Disease_Classification_Deep_learning_project.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "vidyavinodh1982"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "9692187617ef13ca72afffa0b0b432392052aca3"

STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started<<<<<<")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare base model"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started<<<<<<")
    data_ingestion = PrepareBaseModelTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Training"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started<<<<<<")
    model_trainer = ModelTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Evaluation"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started<<<<<<")
    model_evaluation = ModelEvaluationPipeline()
    model_evaluation.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(e)
    raise e