from .resources import *
from impl.services.modules.core.reiform_imageclassificationdataset import *

class Processing_Job():
    def __init__(self, job_json : Dict[str, Any]) -> None:

        self.auto : bool = job_json["auto"]

    def make_dataset(self, body : Dict[str, Any]) -> ReiformICDataSet:
        classes : List[str] = body["classes"]
        
        ds : ReiformICDataSet = ReiformICDataSet(classes)
        ds.from_json(body)

        return ds

    def run_parallel(self, jobs : Callable):
        raise ReiformUnimplementedException("Processing::run_parallel")
