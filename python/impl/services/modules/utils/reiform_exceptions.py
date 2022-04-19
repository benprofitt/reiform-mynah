from __future__ import annotations
import logging, sys
from types import ModuleType

class ReiformLogger(object):

    def __new__(cls: type[ReiformLogger]) -> ReiformLogger:
        if not hasattr(cls, 'instance'):
            log = logging.getLogger()
            log.setLevel(logging.DEBUG)
            stream = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter('mynah-python %(asctime)s %(message)s')
            stream.setFormatter(formatter)
            stream.setLevel(logging.DEBUG)
            log.addHandler(stream)
            cls.instance = super(ReiformLogger, cls).__new__(cls)
            cls.logger : ModuleType = logging

        return cls.instance

class ReiformInfo():
    def __init__(self, message : str="Unimplemented Info Type"):
        ReiformLogger().logger.info(message)

class ReiformWarning():
    def __init__(self, message : str="Unimplemented Warning Type"):
        ReiformLogger().logger.warning(message)

class ReiformException(Exception):
    def __init__(self, message : str="Unimplemented Exception Type"):
        super().__init__(message)

class ReiformUnimplementedException(ReiformException):
    def __init__(self, message : str=""):
        super().__init__("Reiform Unimplemented Method: {}".format(message))

class ReiformClassMethodException(ReiformException):
    def __init__(self, class_name: str, method: str):
        super().__init__("Error in: {}::{}".format(class_name, method))

class ReiformDataSetException(ReiformException):
    def __init__(self, message : str="", method : str="UNKNOWN", dataset_type : str="Unspecified"):
        super().__init__("Error in a ReiformDataSet({})::{} : {}".format(dataset_type, method, message))

class ReiformICDataSetException(ReiformDataSetException):
    def __init__(self, message: str = "", method: str = "UNKNOWN"):
        super().__init__(message, method, "ImageClassification")

class ReiformFileException(ReiformException):
    def __init__(self, message : str="", method : str="UNKNOWN", dataset_type : str="Unspecified"):
        super().__init__("Error in a ReiformFile({})::{} : {}".format(dataset_type, method, message))

class ReiformICFileException(ReiformFileException):
    def __init__(self, message: str = "", method: str = "UNKNOWN"):
        super().__init__(message, method, "ImageClassification")

class ReiformFileSystemException(ReiformException):
    def __init__(self, message: str = "File Not Found"):
        super().__init__(message)

class ReiformDiagnosisException(ReiformException):
    def __init__(self, message: str = ""):
        super().__init__("Error in diagnosis: {}".format(message))

class ReiformCleaningException(ReiformException):
    def __init__(self, message: str = ""):
        super().__init__("Error in data cleaning: {}".format(message))
