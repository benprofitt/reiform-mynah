#!/usr/bin/python3.8
import json
import logging
import sys
from typing import *
from impl.services.modules.utils.progress_logger import ProgressLogger # type: ignore
import impl.services.modules.utils.image_utils as image_utils # type: ignore
import impl.services.image_classification.dataset_processing as processing # type: ignore
from impl.services.modules.utils.progress_logger import ReiformProgressLogger
import argparse
import fileinput
from multiprocessing import Pool

log = logging.getLogger()
log.setLevel(logging.DEBUG)
stream = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('mynah-python %(asctime)s %(message)s')
stream.setFormatter(formatter)
stream.setLevel(logging.DEBUG)
log.addHandler(stream)

logging.getLogger('PIL').setLevel(logging.WARNING)
AVAILABLE_THREADS = 8

def get_impl_version(uuid: str, sock_addr: str) -> str:
    '''Make sure that the module loads, returns version string'''
    logging.info("called get_impl_version()")
    return "{\"status\":0,\"data\":{\"version\":\"0.1.0\"}}"


def start_ic_processing_job(uuid: str, sock_addr: str) -> str:
    '''Start a processing job. See docs/python_api.md'''
    request = json.loads(sys.stdin.read())
    with ProgressLogger(uuid, sock_addr) as plogger:
        logging.info("called start_processing_job()")
        # call impl
        processing_job : processing.DatasetProcessingJob = processing.DatasetProcessingJob(request)
        # Pass in logger to relay progress
        task_results : Dict[str, Any] = processing_job.run_processing_job(plogger)
        
    # response
    return json.dumps({
        "status": 0,
        "data": {
            "dataset": task_results["dataset"],
            "tasks": task_results["tasks"]
        }
    })

def gather_data(obj):
    path = obj["path"]
    return (obj["uuid"], image_utils.get_image_metadata(path))

def get_metadata_for_images(uuid: str, sock_addr: str) -> str:
    '''Get image width, height, channels, mean, std for all images in batch'''

    body = json.loads(sys.stdin.read())

    data = body["images"]

    with Pool(AVAILABLE_THREADS) as p:
        metadatas = p.map(gather_data, data)

    results = {}
    for uuid, metadata in metadatas:
        results[uuid] = metadata

    return json.dumps({
        "status": 0,
        "data": { "images" : results }
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--operation', type=str, required=True)
    parser.add_argument('--ipc-socket-path', type=str, required=True)
    parser.add_argument('--uuid', type=str, required=True)
    args = parser.parse_args()

    # TODO format logger

    #TODO catch exceptions

    try:
        print(locals()[args.operation](args.uuid, args.ipc_socket_path))
    except Exception as e:
        print(json.dumps({
            "status": 1,
            "data": 'unknown exception while executing: {}: {}'.format(args.operation, e)
        }))
