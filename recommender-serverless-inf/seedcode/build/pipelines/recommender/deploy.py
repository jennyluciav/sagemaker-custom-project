import argparse
import logging
from sagemaker import ModelPackage
from time import gmtime, strftime
from sagemaker.serverless import ServerlessInferenceConfig

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def deploy_model(role, sagemaker_session, model_package_arn, endpoint_name):
    logger.info("Starting deployment.")
    model = ModelPackage(role=role, 
                         model_package_arn=model_package_arn, 
                         sagemaker_session=sagemaker_session)
    serverless_config = ServerlessInferenceConfig(
        memory_size_in_mb=2048,
        max_concurrency=5,
    )
    return model.deploy(endpoint_name = endpoint_name, serverless_inference_config=serverless_config)
