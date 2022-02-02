import socket

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

def ipc_test(uuid: str, payload: str, sockaddr: str) -> int:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sockaddr)
    s.send(("{}{}".format(uuid, payload)).encode('utf-8'))
    s.close()
    return len(("{}{}".format(uuid, payload)).encode())
