# Mynah CLI Tool

```shell
Usage of ./mynah-cli:
  -address string
        server address (default "127.0.0.1:8080")
  -jwt string
        auth jwt
  -proto string
        http or https (default "http")
  -task string
        task path (default "task.json")
```

## Task Format
- `tasks` list: array of `task` objects
- `task` object:
  - `task_id`: user-defined identifier for referencing previous steps
  - `task_type`: the type of the task
  - `task_data`: task specific data (depends on `task_type`)

### Upload task
- `file_paths`: paths of files to upload
- `multithread`: whether to upload files in parallel

### Create IC Dataset Task
- `from_existing`: map from existing fileid to classname
- `from_tasks`: reference files uploaded in previous tasks. Sources:
  -  `mynah::upload`
- `local_path_classname_regex`: regex for assigning classname based on local file path
- `dataset_name`: the name to give the new dataset

### Process IC Dataset Task
- `from_existing`: process an existing dataset
- `from_task`: reference a dataset created in a previous task. Sources:
  - `mynah::ic::dataset::create`
- `tasks`: the tasks to run on the dataset
- `poll_frequency`: The frequency in seconds at which to poll for a result

### Get IC Dataset Report
- `from_existing`: request the report from an existing dataset
- `from_task`: request the report from a dataset created in a previous task. Sources:
  - `mynah::ic::dataset::create`
  - `mynah::ic::process`
- `to_file`: write the report to some path

## Examples
- Upload files, create a dataset, run the ic process on the dataset, and download the report
```json
{
  "tasks" : [
    {
      "task_id": "upload_files",
      "task_type": "mynah::upload",
      "task_data": {
        "file_paths" : [
          "docs/test_image.jpg",
          "docs/test_image.jpg",
          "docs/test_image.jpg",
          "docs/test_image.jpg",
          "docs/test_image.jpg"
        ],
        "multithread" : true
      }
    },
    {
      "task_id": "create_dataset",
      "task_type": "mynah::ic::dataset::create",
      "task_data": {
        "from_existing" : {},
        "from_tasks" : [
          "upload_files"
        ],
        "local_path_classname_regex" : "([^\\/]+)\\/[^\\/]+$",
        "dataset_name": "integration dataset 1"
      }
    },
    {
      "task_id": "process",
      "task_type": "mynah::ic::process",
      "task_data": {
        "from_existing" : "",
        "from_task" : "create_dataset",
        "tasks": [
          "ic::diagnose::mislabeled_images"
        ],
        "poll_frequency": 5
      }
    },
    {
      "task_id": "download",
      "task_type": "mynah::ic::report",
      "task_data": {
        "from_existing" : "",
        "from_task" : "create_dataset",
        "to_file": "tmp.json"
      }
    }
  ]
}
```