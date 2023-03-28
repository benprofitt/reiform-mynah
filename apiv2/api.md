# API Version 2

## Types

### Enums
- `MynahDatasetType`
```json
[
  "image_classification"
]
```
- `MynahICProcessTaskType`
```json
[
  "ic::diagnose::mislabeled_images",
  "ic::correct::mislabeled_images",
  "ic::diagnose::class_splitting",
  "ic::correct::class_splitting"
]
```

### First-class Types
- `MynahUser`
```json
{
  "user_id": "",
  "name_first": "",
  "name_last": ""
} 
```
- `MynahDataset`
```json
{
  "dataset_id": "",
  "dataset_name": "",
  "date_created": 0,
  "date_modified": 0,
  "dataset_type": MynahDatasetType,
  "created_by": ""
} 
```
- `MynahICDatasetVersion`
```json
{
  "dataset_version_id": "",
  "dataset_id": "",
  "ancestor_id": "",
  "date_created": 0,
  "mean": [0.0],
  "std_dev": [0.0],
  "task_data": [
    MynahICProcessTaskData
  ],
  "created_by": ""
} 
```
- `MynahICProcessTaskData`
```json
{
  "type": MynahICProcessTaskType,
  "metadata": MynahICProcessTaskMetadata
} 
```
- `MynahICProcessTaskMetadata`
```json
MynahICProcessTaskDiagnoseMislabeledImagesMetadata | MynahICProcessTaskCorrectMislabeledImagesMetadata | MynahICProcessTaskDiagnoseClassSplittingMetadata | MynahICProcessTaskCorrectClassSplittingMetadata
```
- `MynahICProcessTaskDiagnoseMislabeledImagesMetadata`
```json
{
  "outliers": [""]
} 
```
- `MynahICProcessTaskCorrectMislabeledImagesMetadata`
```json
{
  "removed": [""],
  "corrected": [""]
} 
```
- `MynahICProcessTaskDiagnoseClassSplittingMetadata`
```json
{
  "predicted_class_splits": {
    "some_class_name": [[""]]
  }
} 
```
- `MynahICProcessTaskCorrectClassSplittingMetadata`
```json
{
  "actual_class_splits": {
    "some_class_name": {
      "some_class_name": [""]
    }
  }
} 
```
- `MynahDatasetVersionRef`
```json
{
  "dataset_version_id": "",
  "ancestor_id": "",
} 
```
- `MynahFile`
```json
{
  "file_id": "",
  "name": "",
  "date_created": 0,
  "mime_type": ""
} 
```
- `MynahICDatasetVersionFile`
```json
{
  "file_id": "",
  "dataset_version_id": "",
  "class": "",
  "confidence_vectors": [[0.0]],
  "projections" : {
    "projection_label_full_embedding_concatenation": [0.0],
    "projection_label_reduced_embedding": [0.0],
    "projection_label_reduced_embedding_per_class": [0.0],
    "projection_label_2d_per_class": [0.0],
    "projection_label_2d": [0.0],
  },
  "mean": [0.0],
  "std_dev": [0.0]
} 
```
- `MynahDatasetReport`
```json
{
  "report_id": "",
  "dataset_version_id": "",
  "date_created": 0,
  "created_by": ""
} 
```
- `MynahICDatasetVersionReportData`
```json
{
  "report_id": "",
  "contents": {
    "type": MynahICProcessTaskType,
    "metadata": MynahICProcessTaskReportMetadata,
  }
} 
```
- `MynahICProcessTaskReportMetadata`
```json
MynahICProcessTaskDiagnoseMislabeledImagesReport | MynahICProcessTaskCorrectMislabeledImagesReport | MynahICProcessTaskDiagnoseClassSplittingReport | MynahICProcessTaskCorrectClassSplittingReport
```
- `MynahICProcessTaskDiagnoseMislabeledImagesReport`
```json
{
  "class_label_errors": {
    "some_class_name": {
      "mislabeled": [""],
      "correct": [""]
    }
  }
} 
```
- `MynahICProcessTaskCorrectMislabeledImagesReport`
```json
{
  "class_label_errors": {
    "some_class_name": {
      "mislabeled_corrected": [""],
      "mislabeled_removed": [""],
      "unchanged": [""]
    }
  }
} 
```
- `MynahICProcessTaskDiagnoseClassSplittingReport`
```json
{
  "classes_splitting": {
    "some_class_name": {
      "predicted_classes_count": 0
    }
  }
} 
```
- `MynahICProcessTaskCorrectClassSplittingReport`
```json
{
  "classes_splitting": {
    "some_class_name": {
      "new_classes": [""]
    }
  }
} 
```
- `MynahICDatasetVersionReportPoint`
```json
{
  "report_id": "",
  "file_id": "",
  "class": "",
  "point": [0.0]
} 
```

### Other Types
- `Paginated`
```json
{
  "page": 1,
  "page_size": 20,
  "total": 150,
  "contents": [
    
  ]
}
```

## Pagination
- Pages are 0-indexed
- All paginated endpoints take _optional_ query parameters `page` (defaults to 0), and `page_size` (defaults to a settings value)
- Examples
  - The first page of datasets `/api/v2/dataset/list`
  - The second page of datasets `/api/v2/dataset/list?page=1`
  - The first page of datasets with page size 10 `/api/v2/dataset/list?page_size=10`
  - The third page of datasets with page size 15 `/api/v2/dataset/list?page=2&page_size=15`

## Endpoints

### User
#### User Creation
- `POST` `/api/v2/user/create`
    - Request:
    ```json
    {
      "name_first": "",
      "name_last": ""
    }
    ```
    - Response: `MynahUser`

### File

#### File Upload
- `POST` `/api/v2/dataset/{dataset_id}/version/{version_id}/upload`
    - Request: multipart form with `file` containing contents, `class` containing initial class assignment
    - Response: `MynahFile`

#### Requesting the raw contents of a file
- `GET` `/api/v2/raw/{file_id}`

#### Requesting file data
- `GET` `/api/v2/file/{file_id}`
    - Response: `MynahFile`

### Dataset

#### Dataset creation
- `POST` `/api/v2/dataset/create`
    - Request:
    ```json
    {
      "dataset_name": "",
      "dataset_type": MynahDatasetType
    }
    ```
    - Response: `MynahDataset`

#### Get a Dataset
- `GET` `/api/v2/dataset/{dataset_id}`
  - Response: `MynahDataset`

#### List Datasets
- `GET` `/api/v2/dataset/list`
    - Response: `Paginated[MynahDataset]`

#### Get the dataset versions
- `GET` `/api/v2/dataset/{dataset_id}/version/refs`
  - Response: `[MynahDatasetVersionRef]`

#### Get a dataset version
- `GET` `/api/v2/dataset/{dataset_id}/version/{version_id}`
    - Response: `MynahICDatasetVersion` (Others in the future)

#### List dataset versions
- `GET` `/api/v2/dataset/{dataset_id}/version/list`
    - Response: `Paginated[MynahICDatasetVersion]` (Others in the future)

#### Get report for dataset version
- `GET` `/api/v2/dataset/{dataset_id}/version/{version_id}/report/{report_id}`
  - Response: `MynahDatasetReport`

#### List reports for a dataset version
- `GET` `/api/v2/dataset/{dataset_id}/version/{version_id}/report/list`
  - Response: `Paginated[MynahDatasetReport]`

#### Get the report contents for a dataset version
- `GET` `/api/v2/dataset/{dataset_id}/version/{version_id}/report/{report_id}/contents`
  - Response: `MynahICDatasetReportContents` (Others in the future)

#### List files in a dataset version
- `GET` `/api/v2/dataset/{dataset_id}/version/{version_id}/file/list`
  - Response: `Paginated[MynahFile]`

#### Dataset file class assignment (NOT CURRENTLY SUPPORTED)
- `GET` `/api/v2/dataset/{dataset_id}/version/{version_id}/file/classes`
  - Request:
    ```json
    {
      "assignments" : {
        "file_id1": "classname",
        "file_id2": "classname",
      }
    }
    ```


### Operations

#### Starting a processing job
- `POST` `/api/v2/dataset/{dataset_id}/version/{version_id}/process`
  - Request:
    ```json
    {
      "tasks" [
        MynahICProcessTaskType
      ]
    }
    ```
