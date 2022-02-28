from .job_processing import *

class Diagnosis_Job(Processing_Job):
    def __init__(self, job_json: Dict[str, Any]) -> None:
        super().__init__(job_json)

    def run_diagnosis(self, logger : ProgressLogger):
        if self.auto:
            raise ReiformDiagnosisException("need to implement auto diagnosis")
        else:
            task : Dict[str, Any]
            results : List[Dict[str, Any]] = {}
            for i, task in enumerate(self.tasks):
                name : str = task["name"]

                try:
                    results_json : Dict[str: Any] = eval("self.{}()".format(name))
                    results.append({"name" : name, "datasets": results_json})
                    logger.write(json.dumps({"completed": name, "progress" : i/len(self.tasks)}))
                except:
                    raise ReiformDiagnosisException("diagnosis category {} does not exist".format(name))

        return results

    def mislabeled_images(self):

        projection : ReiformICDataSet = vae_projection(self.data.copy(), 2)

        inliers, outliers = find_outliers_loda(projection)

        return  {
                    "inliers": inliers.to_json(), 
                    "outliers": outliers.to_json()
                }
