from .job_processing import *

class Cleaning_Job(Processing_Job):
    def __init__(self, job_json: Dict[str, Any]) -> None:
        super().__init__(job_json)