import json
import logging
import sys
from impl.services.modules.utils.progress_logger import ProgressLogger # type: ignore
import impl.services.modules.utils.image_utils as image_utils # type: ignore

log = logging.getLogger()
log.setLevel(logging.DEBUG)
stream = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('mynah-python %(asctime)s %(message)s')
stream.setFormatter(formatter)
stream.setLevel(logging.DEBUG)
log.addHandler(stream)

def get_impl_version(uuid: str, request_str: str, sock_addr: str) -> str:
    '''Make sure that the module loads, returns version string'''
    logging.info("INFO  called get_impl_version()")
    return "{\"version\": \"0.1.0\"}"

def start_diagnosis_job(uuid: str, request_str: str, sock_addr: str) -> str:
    '''Start a diagnosis job. See docs/python_api.md'''
    request = json.loads(request_str)
    with ProgressLogger(uuid, sock_addr) as plogger:
        logging.info("INFO  called start_diagnosis_job()")
        # TODO call impl

    # TODO response
    return "{}"

def get_image_metadata(path: str) -> str:
    '''Retrieve the image width, height, and channels'''
    return json.dumps(image_utils.get_image_metadata(path))