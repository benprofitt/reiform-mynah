from impl.services.modules.ic_inference.inference_interface import process_dataset
from impl.services.modules.utils.progress_logger import ReiformProgressLogger
from .resources import *
from .job_processing import Model_Processing_Job

class InferenceJob(Model_Processing_Job):

    def __init__(self, job_json: Dict[str, Any]) -> None:
        super().__init__(job_json)

        self.model, self.transform = self.rebuild_model_and_transform(job_json[MODEL_METADATA])
        self.size : List[int] = job_json[MODEL_METADATA][IMAGE_DIMS]
        self.size.append(3)

    def rebuild_model_and_transform(self, model_json: Dict[str, Any]) -> nn.Module:

        model = self.load_model(model_json[PATH_TO_MODEL], 
                                     self.create_model(self.dataset.classes(), model_json[MODEL]))


        size : int = model_json[IMAGE_DIMS]
        mean = model_json[MEAN]
        std = model_json[STD]

        transform : torchvision.transforms.Compose = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        return model, transform

    def make_result(self) -> Dict[str, Any]:
        body : Dict[str, Any] = {"dataset" : self.dataset.serialize()}
        return body

    def run_processing_job(self, logger : ReiformProgressLogger) -> Dict[str, Any]:

        # process_dataset
        self.dataset = process_dataset(self.dataset, self.model, self.transform, self.size)

        logger.write("finished inference; serializing results.")

        return self.make_result()