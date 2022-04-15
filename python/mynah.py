import json
import logging
import sys
from typing import *
from impl.services.modules.utils.progress_logger import ProgressLogger # type: ignore
import impl.services.modules.utils.image_utils as image_utils # type: ignore
import impl.services.image_classification.diagnosis_job as diagnosis # type: ignore
import impl.services.image_classification.cleaning_job as cleaning # type: ignore


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
    logging.info("INFO  called get_impl_version()")
    return "{\"version\": \"0.1.0\"}"

def start_diagnosis_job(uuid: str, request_str: str, sock_addr: str) -> str:
    '''Start a diagnosis job. See docs/python_api.md'''
    request = json.loads(request_str)
    with ProgressLogger(uuid, sock_addr) as plogger:
        logging.info("INFO  called start_diagnosis_job()")
        # call impl
        diagnosis_job : diagnosis.Diagnosis_Job = diagnosis.Diagnosis_Job(request)

        # Pass in logger to relay progress
        task_results : List[Dict[str, Any]] = diagnosis_job.run_diagnosis(plogger)

    # response
    return json.dumps({
                        "dataset_uuid" : request["dataset_uuid"],
                        "tasks" : task_results
                      })


def start_cleaning_job(uuid: str, request_str: str, sock_addr: str) -> str:
    '''Start a cleaning job. See docs/python_api.md'''
    request = json.loads(request_str)
    with ProgressLogger(uuid, sock_addr) as plogger:
        logging.info("INFO  called start_diagnosis_job()")
        # call impl
        cleaning_job : cleaning.Cleaning_Job = cleaning.Cleaning_Job(request)

        # Pass in logger to relay progress
        cleaning_results : List[Dict[str, Any]] = cleaning_job.run_cleaning(plogger)

    # response
    return json.dumps({
                        "dataset_uuid" : request["dataset_uuid"],
                        "datasets" : cleaning_results
                      })

def get_image_metadata(uuid: str, request_str: str, sock_addr: str) -> str:
    '''Retrieve the image width, height, and channels'''
    request = json.loads(request_str)
    return json.dumps(image_utils.get_image_metadata(request['path']))