

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
        super().__init__("Error in a ReiformDataSet({})::{} : {}"(dataset_type, method, message))

class ReiformICDataSetException(ReiformDataSetException):
    def __init__(self, message: str = "", method: str = "UNKNOWN"):
        super().__init__(message, method, "ImageClassification")

class ReiformFileException(ReiformException):
    def __init__(self, message : str="", method : str="UNKNOWN", dataset_type : str="Unspecified"):
        super().__init__("Error in a ReiformFile({})::{} : {}"(dataset_type, method, message))

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
