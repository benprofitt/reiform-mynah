from impl.services.modules.class_splitting.correction import split_dataset
from impl.services.modules.class_splitting.detection import detect_split_need
from impl.services.modules.core.reiform_imageclassificationdataset import *
from impl.services.image_classification.job_processing import Processing_Job
from .resources import *

class DatasetProcessingJob(Processing_Job):
    def __init__(self, job_json : Dict[str, Any]) -> None:
        super().__init__(job_json)
        self.previous_results : Dict[str, Dict[str, Any]] = self.make_previous_results(job_json["previous_results"])

        self.tasks_requested : List[str] = self.order_tasks(list([v["type"] for v in job_json["tasks"]]))

        self.config : Dict[str, Any] = job_json["config_params"]

        # Outgoing data
        self.results : Dict[str, Dict[str, Any]] = {}

    def make_result(self):
        body : Dict[str, Any] = {"dataset" : self.dataset.serialize(), "tasks" : []}
        task_packages : list[dict] = []

        for task in self.tasks_requested:
            package = {"type" : task, 
                       "metadata" : self.results[task]}
            task_packages.append(package)

        body["tasks"] = task_packages

        return body

    def make_previous_results(self, old_results : List[Dict[str, Any]]):
        results : Dict[str, Dict[str, Any]] = {}

        for d in old_results:
            data_type, action, task = d["type"].split("::")
            if data_type != "ic":
                ReiformUnimplementedException("Reiform only offers IC capability at this time.")

            metadata = d["Metadata"]
            if task not in results:
                results[task] = {}
            results[task][action] = metadata
            
        return results


    def order_tasks(self, tasks_list : List[str]):
        # Standard order for tasks to be most effective
        std_order = ["class_splitting", "mislabeled_images"]

        self.raw_tasks = copy.copy(tasks_list)
        task_list : List[List[str]] = [t.split("::") for t in tasks_list]

        task_types : Dict[str, Dict[str, bool]] = {}

        # If we're correcting anything, that needs to happen _before_ any detection that isn't also corrected
        data_type : str
        action_type : str
        task_name : str
        for data_type, action_type, task_name in task_list:
            if data_type != "ic":
                ReiformUnimplementedException("Reiform only offers IC capability at this time.")
            if task_name not in task_types:
                task_types[task_name] = {}
            task_types[task_name][action_type] = True

        actions_to_order : List[str] = []
        end_of_actions_to_order : List[str] = []

        for task in std_order:
            if task not in task_types:
                continue

            actions = task_types[task]
            if "diagnose" not in actions:
                if task not in self.previous_results or "diagnose" not in self.previous_results[task]:
                    ReiformWarning("No previous diagnosis before requesting cleaning. Diagnosis will be performed.")
                    actions_to_order.append("{}::{}::{}".format("ic", "diagnose", task))
            else:
                if "correct" not in actions:
                    # If we're diagnosing and not correcting it should come after all the correction tasks
                    end_of_actions_to_order.append("{}::{}::{}".format("ic", "diagnose", task))
                else:
                    actions_to_order.append("{}::{}::{}".format("ic", "diagnose", task))
            
            if "correct" in actions:
                actions_to_order.append("{}::{}::{}".format("ic", "correct", task))

        actions_to_order += end_of_actions_to_order
        
        return actions_to_order

    def run_processing_job(self, logger):

        models_path = self.config["models_path"]
        embedding_models_path = "{}/{}".format(models_path, EMBEDDING_MODEL_NAME)

        prev_result_names = [v.split("::")[2] for v in list(self.previous_results.keys())]
        valid_embeddings = any([v in prev_result_names for v in ["mislabeled_images", "class_splitting"]])

        for i, task in enumerate(self.tasks_requested):
            logger.write("Starting {} ({}/{})".format(task, i+1, len(self.tasks_requested)))
            ReiformInfo("Starting {} ({}/{})".format(task, i+1, len(self.tasks_requested)))
            _, action, name = task.split("::")

            if action == "diagnose":
                ReiformInfo("Starting diagnosis")
                if name == "mislabeled_images":
                    if not valid_embeddings:
                        ReiformInfo("Creating embeddings")
                        create_dataset_embedding(self.dataset, embedding_models_path)
                        valid_embeddings = True

                    _, outliers = find_outlier_consensus(self.dataset)
                    
                    self.results[task] = {"outliers" : [f.get_uuid() for f in outliers.all_files()]}
                    self.previous_results[task] = self.results[task]

                if name == "class_splitting":
                    if not valid_embeddings:
                        create_dataset_embedding(self.dataset, embedding_models_path)
                        valid_embeddings = True
                    _, clusters = detect_split_need(self.dataset)
                    self.results[task] = {"predicted_class_splits" : clusters}
                    self.previous_results[task] = self.results[task]

            elif action == "correct":
                if name == "mislabeled_images":

                    if not valid_embeddings:
                        create_dataset_embedding(self.dataset, embedding_models_path)

                    to_correct = self.previous_results["ic::diagnose::mislabeled_images"]["outliers"]
                    
                    outliers = self.dataset.dataset_from_uuids(to_correct)
                    as_keys = set(to_correct)
                    
                    inlier_ids = [f.get_name() for f in self.dataset.all_files() if f.get_uuid() not in as_keys]
                    ReiformInfo("Classes: {}".format(self.dataset.classes()))
                    inliers = self.dataset.dataset_from_names(inlier_ids)
                    inliers.uuid = self.dataset.uuid
                    
                    self.dataset, outliers, corrected = iterative_reinjection_label_correction(5, inliers, outliers)
                    
                    self.results[task] = {"removed" : [f.get_uuid() for f in outliers.all_files()], 
                                            "corrected" : [f.get_uuid() for f in corrected.all_files()]}
                    
                    create_dataset_embedding(self.dataset, embedding_models_path)
                    valid_embeddings = True
                    self.dataset._mean_and_std_dev()

                if name == "class_splitting":
                    if not valid_embeddings:
                        create_dataset_embedding(self.dataset, embedding_models_path)
                    
                    to_split = self.previous_results[task]["predicted_class_splits"]

                    self.dataset, actual_class_splits = split_dataset(self.dataset, list(to_split.keys()))

                    self.results[task] = {"actual_class_splits" : actual_class_splits}

                    create_dataset_embedding(self.dataset, embedding_models_path)
                    valid_embeddings = True

            logger.write("Finished {} ({}/{})".format(task, i+1, len(self.tasks_requested)))

        return self.make_result()

    def run_parallel(self, jobs : Callable):
        raise ReiformUnimplementedException("Processing::run_parallel")
