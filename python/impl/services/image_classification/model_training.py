from impl.services.modules.ic_training.training_interface import TrainingSpecifications, train_ic_model
from impl.services.modules.utils.progress_logger import ReiformProgressLogger
from .resources import *
from .job_processing import Processing_Job

class TrainingJob(Processing_Job):

    def __init__(self, job_json: Dict[str, Any]) -> None:
        super().__init__(job_json)
        self.mean = self.dataset.get_mean()
        self.std = self.dataset.get_std_dev()

        self.config = job_json["config_params"]
        self.save_path = self.config["model_save_path"]

        self.train_loss : List[float] = []
        self.test_losses : List[List[float]] = []
        self.filename_order : List[str] = []

    def make_result(self) -> Dict[str, Any]:
        body : Dict[str, Any] = {}
        
        try:
            torch.save(self.model.state_dict(), self.save_path)
            body["model_saved"] = True
        except:
            body["model_saved"] = False

        body["config_params"] = self.config
        body["avg_training_loss"] = self.train_loss
        # This is useful for showing users which images have high losses
        body["all_file_test_loss"] = self.test_losses
        body["ordered_fileids"] = self.filename_order

        body["model_metadata"] = self.make_metadata()

        return body

    def make_metadata(self):
        return {
            "model" : self.config["model"],
            "image_dims" : [299, 299], # for now this is constant, but provides flexibility
            "type" : "mynah::IC::model",
            "classes" : self.dataset.classes(),
            "mean" : self.mean,
            "std" : self.std,
            "path_to_model" : self.save_path
        }

    def save_model(self, model : torch.nn.Module, channels : int, 
               size : int, mean : List[float], std : List[float]) -> None:
        # Save the trained model along with metadata
        # I think that this is a potential option to ensure the model can be loaded and 
        # used properly later.
        json_body = {
            CHANNELS : channels,
            SIZE : size, # input edge size
            MEAN : mean,
            STD : std
        }

        local_path = self.save_path

        Path("/".join(local_path.split("/")[0:-1])).mkdir(exist_ok=True, parents=True)

        model_path = "{}{}".format(local_path, "_model.pt")
        metadata_path = "{}{}".format(local_path, "_metadata.json")

        # Save the model and the json
        torch.save(model.state_dict(), model_path)
        with open(metadata_path, 'w') as fh:
            fh.write(json.dumps(json_body, indent=2))

        return

    def run_processing_job(self, logger : ReiformProgressLogger) -> Dict[str, Any]:

        MODEL_LIST = ["resnet50"]#, "resnet18", "densenet202"]
        MODEL_MAP = {
            "resnet18" : "AutoResnet18",
            "resnet50" : "AutoResnet",
            "densenet202" : "AutoDensenet202"
        }

        model_name = self.config["model"]
        if model_name not in MODEL_LIST:
            raise ReiformTrainingException("Model not available")

        model_classname : str = MODEL_MAP[model_name]
        target_classes : int = len(self.dataset.classes())

        self.model = eval("{}({})".format(model_classname, target_classes))

        min_epochs = self.config["min_epochs"]
        max_epochs = self.config["max_epochs"]

        loss_epsilon = self.config["loss_epsilon"]
        batch_size = self.config["batch_size"]

        train_test_split = self.config["train_test_ratio"]

        mean=self.mean
        std=self.std

        transform : torchvision.transforms.Compose = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        optimizer = get_optimizer(self.model)

        training_spec = TrainingSpecifications(max_epochs, min_epochs, optimizer, 
                                            loss_epsilon, batch_size, train_test_split,
                                            transform)

        self.model, self.train_loss, self.test_losses, self.filename_order = train_ic_model(self.dataset, self.model, training_spec, logger)

        return self.make_result()