import json
import logging
import sys
from typing import *
from impl.services.modules.utils.progress_logger import ProgressLogger # type: ignore
import impl.services.modules.utils.image_utils as image_utils # type: ignore
import impl.services.image_classification.dataset_processing as processing # type: ignore
import impl.services.image_classification.model_training as training # type: ignore

log = logging.getLogger()
log.setLevel(logging.DEBUG)
stream = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('mynah-python %(asctime)s %(message)s')
stream.setFormatter(formatter)
stream.setLevel(logging.DEBUG)
log.addHandler(stream)

logging.getLogger('PIL').setLevel(logging.WARNING)


def get_impl_version(uuid: str, request_str: str, sock_addr: str) -> str:
    '''Make sure that the module loads, returns version string'''
    logging.info("called get_impl_version()")
    return "{\"version\": \"0.1.0\"}"


def start_ic_processing_job(uuid: str, request_str: str, sock_addr: str) -> str:
    '''Start a processing job. See docs/python_api.md'''
    request = json.loads(request_str)
    with ProgressLogger(uuid, sock_addr) as plogger:
        logging.info("called start_processing_job()")
        # call impl
        processing_job : processing.DatasetProcessingJob = processing.DatasetProcessingJob(request)

        # Pass in logger to relay progress
        task_results : Dict[str, Any] = processing_job.run_processing_job(plogger)

    # response
    return json.dumps({
                        "dataset_uuid": request["dataset_uuid"],
                        "tasks": task_results
                      })

def start_ic_training_job(uuid : str, request_str : str, sock_addr: str) -> str:
    '''Start an IC training job. See docs/python_api.md'''
    request = json.loads(request_str)
    with ProgressLogger(uuid, sock_addr) as plogger:
        logging.info("called start_training_job()")
    
        # call impl
        training_job : training.TrainingJob = training.TrainingJob(request)

        # Pass in logger to relay progress
        training_results : Dict[str, Any] = training_job.run_processing_job(plogger)

    # response
    return json.dumps({
                        "model_uuid": request["model_uuid"],
                        "results": training_results
                      })

def get_image_metadata(uuid: str, request_str: str, sock_addr: str) -> str:
    '''Retrieve the image width, height, and channels'''
    body = json.loads(request_str)
    path = body['path']
    return json.dumps(image_utils.get_image_metadata(path))
