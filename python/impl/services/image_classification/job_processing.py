from .resources import *

class Processing_Job():
    def __init__(self, job_json : Dict[str, Any]) -> None:

        # Incoming data
        self.dataset : ReiformICDataSet = self.make_dataset(job_json["dataset"])
        
    def make_result(self) -> Dict[str, Any]:
        body : Dict[str, Any] = {}
        return body

    def make_dataset(self, body : Dict[str, Any]) -> ReiformICDataSet:
        classes : List[str] = body["classes"]
        
        ds : ReiformICDataSet = ReiformICDataSet(classes)
        ds.from_json(body)

        return ds

    def run_processing_job(self, logger):
        raise ReiformUnimplementedException("ProcessingJob::run_processing_job")