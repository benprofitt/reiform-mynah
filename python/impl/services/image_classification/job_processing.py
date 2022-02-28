from .resources import *

class Processing_Job():
    def __init__(self, job_json : Dict[str, Any]) -> None:

        dataset : Dict[str, Any] = job_json["dataset"]
        self.auto : bool = job_json["auto"]
        if not self.auto:
            self.tasks : List[Dict[str, Any]] = job_json["tasks"]

        self.data : ReiformICDataSet = self.make_dataset(dataset)

    def make_dataset(self, body : Dict[str, Any]) -> ReiformICDataSet:
        classes : List[str] = body["classes"]
        files : Dict[str, Dict[str, Dict[str, Any]]] = body["class_files"]
        
        ds : ReiformICDataSet = ReiformICDataSet(classes)
        ds.from_json(files)

        return ds

    def run_parallel(self, jobs : Callable):
        raise ReiformUnimplementedException("Processing::run_parallel")
