import socket
import json
from impl.services.modules.utils.progress_logger import ProgressLogger # type: ignore
# from mynah import * # type: ignore
import impl.services.modules.utils.image_utils as image_utils
from string import Template

from impl.services.modules.utils.reiform_exceptions import ReiformInfo


def test0(_int : int, _float : float) -> str:
    if (_int != 3) or (_float != 1.2):
        return "None"
    # res = ""
    # for s in list_strings:
    #     res += s
    return "ab"

def test1(str1 : str, str2 : str) -> str:
    return str1 + str2

def test2() -> None:
    return None

def test3(s : str, i : int) -> int:
    return len(s) + i

def test4() -> int:
    raise ValueError("python test exception")

def image_metadata_test():
    path : str = "impl/test/test_data_mnist/0/img_322.jpg"
    try:
        ReiformInfo(get_image_metadata(path))
    except:
        raise FileNotFoundError("File {} not found. Run the test from inside python".format(path))

def ipc_test(uuid: str, payload: str, sockaddr: str) -> str:
    #parse payload as json
    contents = json.loads(payload)
    with ProgressLogger(uuid, sockaddr) as plogger:
        plogger.write(contents['msg'])

    return '{"msg": "%s"}' % contents['msg']

def start_ic_processing_job(uuid: str, request_str: str, sock_addr: str) -> str:
    # TODO check that the request body is correct
    contents = json.loads(request_str)
    dataset_uuid = contents['dataset']['uuid']
    t = Template('''{
    "dataset": {
      "uuid": "${uuid}",
      "classes" : ["class1", "class2"],
      "mean": [0.3, 0.4, 0.1],
      "std_dev": [0.1, 0.12, 0.03],
      "class_files" : {
        "class1" : {
          "/tmp/fileuuid1" : {
            "uuid": "fileuuid1",
            "current_class": "class1",
            "projections": {},
            "confidence_vectors": [[1.0, 2.0]],
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03]
          },
          "/tmp/fileuuid2" : {
            "uuid": "fileuuid2",
            "current_class": "class2",
            "projections": {},
            "confidence_vectors": [[1.0, 2.0]],
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03]
          }
        },
        "class2" : {
          "/tmp/fileuuid3" : {
            "uuid": "fileuuid3",
            "current_class": "class2",
            "projections": {},
            "confidence_vectors": [[1.0, 2.0]],
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03]
          },
          "/tmp/fileuuid4" : {
            "uuid": "fileuuid4",
            "current_class": "class2",
            "projections": {},
            "confidence_vectors": [[1.0, 2.0]],
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03]
          }
        }
      }
    },
    "tasks": [ 
      {
        "type" : "ic::diagnose::mislabeled_images",
        "metadata": {
          "outliers" : ["fileuuid1", "fileuuid3"]
        }
      },
      {
        "type" : "ic::correct::lighting_conditions",
        "metadata": {
          "removed" : ["fileuuid1", "fileuuid2"],
          "corrected" : ["fileuuid3"]
        }
      }
    ]
  }''')
    return t.substitute(uuid=dataset_uuid)
