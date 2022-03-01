from .job_processing import *

class Cleaning_Job(Processing_Job):
    def __init__(self, job_json: Dict[str, Any]) -> None:
        super().__init__(job_json)
        
        self.task_data : Dict[str : Dict[str, ReiformICDataSet]] = {}

        for task in job_json["tasks"]:
            task_name : str = task["name"]
            self.task_data[task_name] = {} 
            for name, ds in task["datasets"].items():
                self.task_data[name] = self.make_dataset(ds)

        
    def order_tasks(self) -> List[str]:
        ordered_tasks : List[str] = []
        on_end : List[str] = []
        for task in self.task_data:
            if task == "mislabeled_images":
                on_end.append(task)
            ordered_tasks.append(task)
        
        for task in on_end:
            ordered_tasks.append(task)
        return ordered_tasks

    def run_cleaning(self, logger : ProgressLogger):
        if self.auto:
            raise ReiformCleaningException("need to implement auto cleaning")
        else:
            
            results : List[Dict[str, Any]] = {}
            ordered_tasks : List[str] = self.order_tasks()

            for i, name in enumerate(ordered_tasks):
                
                try:
                    inliers : ReiformICDataSet = self.task_data[name]["inliers"]
                    outliers : ReiformICDataSet = self.task_data[name]["outliers"]

                    if name == "mislabeled_images":

                        for res in results:
                            rem  = res["datasets"]["removed"]
                            inliers = inliers.set_minus(rem)
                            outliers.merge_in(rem)
                        
                    datasets = {"inliers" : inliers, "outliers" : outliers}

                    results_json : Dict[str: Any] = eval("self.{}(datasets)".format(name))
                    results.append({"name" : name, "datasets": results_json})
                    logger.write(json.dumps({"completed": name, "progress" : i/len(self.task_data)}))
                except:
                    raise ReiformCleaningException("cleaning category {} does not exist".format(name))

            corrected : ReiformICDataSet = ReiformICDataSet(inliers.classes())
            if "mislabeled_images" not in ordered_tasks:
                removed : ReiformICDataSet = results[0]["datasets"]["removed"]
                for res in results:
                    corr = res["datasets"]["corrected"]
                    rem  = res["datasets"]["removed"]
                    corrected.merge_in(corr)
                    removed = removed.intersection(rem)
            else:
                corrected = inliers
                removed = outliers

        return {"corrected" : corrected.to_json(), "removed" : removed.to_json()}

    def mislabeled_images(self, diagnosed_datasets : Dict[str: ReiformICDataSet]):

        inliers, outliers, corrected = iterative_reinjection_label_correction(1, diagnosed_datasets["inliers"], diagnosed_datasets["outliers"])

        return  {"corrected": inliers.merge(corrected), "removed": outliers}

