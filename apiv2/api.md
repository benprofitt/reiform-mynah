# API Version 2

## Types

### Enums
- `MynahDatasetType`
```json
[
  "image_classification"
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
} 
```
- `MynahICDatasetVersion`
```json
{
  "dataset_version_id": "",
  "version_index": 0,
  "date_created": 0,
  "mean": [0.0],
  "std_dev": [0.0],
  "task_data": [
    TODO
  ]
} 
```
- `MynahDatasetReport`
```json
{
  "report_id": "",
  "date_created": 0,
  "created_by": ""
} 
```
- `MynahICDatasetReportContents`
```json
{
  "report_id": "",
  TODO
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

- `MynahDatasetVersionRef`
```json
{
  "dataset_version_id": "",
  "ancestor_id": ""
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
    - Request: multipart form with `file` containing contents
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

### Operations

#### Starting a processing job
- `POST` `/api/v2/dataset/{dataset_id}/process`
  - Request:
    ```json
    {
      "ancestor_version_id": "",
      "tasks" [
        TODO
      ]
    }
    ```
