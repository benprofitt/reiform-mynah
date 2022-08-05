import socket

class ReiformProgressLogger():

    def __init__(self):
        self.value = 1

    def write(self, msg: str) -> None:
        # raise Exception()
        print()

class ProgressLogger(ReiformProgressLogger):
    """
    Writes json or other data to
    the websocket server that frontend clients
    can connect to
    """
    def __init__(self, uuid: str, sock_addr: str):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock_addr = sock_addr
        self._uuid = uuid

    def __enter__(self) -> 'ProgressLogger':
        '''Connect to the socket'''
        self._sock.connect(self._sock_addr)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        '''Close the socket'''
        self._sock.close()

    def write(self, msg: str) -> None:
        '''Write a message to the socket'''
        self._sock.send(("{0: <36}{1}".format(self._uuid, msg)).encode('utf-8'))

class TestLogger(ReiformProgressLogger):
    """
    Writes json or other data to
    std out for testing only
    """
    def __init__(self):
        print("Test logger init")

    def write(self, msg: str) -> None:
        '''Write a message to the socket'''
        print(msg)