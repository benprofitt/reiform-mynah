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

class Model_Processing_Job(Processing_Job):

    def __init__(self, job_json: Dict[str, Any]) -> None:
        super().__init__(job_json)

    
    def create_model(self, class_list : List[str], model_name : str) -> nn.Module:

        MODEL_LIST = ["resnet50"]#, "resnet18", "densenet202"]
        MODEL_MAP = {
            "resnet18" : "AutoResnet18",
            "resnet50" : "AutoResnet",
            "densenet202" : "AutoDensenet202"
        }

        if model_name not in MODEL_LIST:
            raise ReiformTrainingException("Model not available")

        model_classname : str = MODEL_MAP[model_name]
        target_classes : int = len(class_list)

        model = eval("{}({})".format(model_classname, target_classes))
        
        return model

    def load_model(self, path : str, model : nn.Module) -> nn.Module:

        # Load the state dict
        model.load_state_dict(torch.load(path))
        model.eval()
        
        return model