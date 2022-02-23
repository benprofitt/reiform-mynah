import socket
import json
from impl.services.modules.utils.progress_logger import ProgressLogger # type: ignore
from mynah import * # type: ignore

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
        print(get_image_metadata(path))
    except:
        raise FileNotFoundError("File {} not found. Run the test from inside python".format(path))

def ipc_test(uuid: str, payload: str, sockaddr: str) -> str:
    #parse payload as json
    contents = json.loads(payload)
    with ProgressLogger(uuid, sockaddr) as plogger:
        plogger.write(contents['msg'])

    return '{"msg": "%s"}' % contents['msg']

def start_diagnosis_job(uuid: str, request_str: str, sock_addr: str) -> str:
    # TODO check that the request body is correct
    return "{}"
