# API Endpoint Specification

## Repeated Types
### `MynahUser`
  ```json
  {
    "uuid" : "",
    "name_first" : "",
    "name_last" : ""
  }
  ```
### `MynahICDataset`
  ```json
  {
    "uuid" : "",
    "owner_uuid": "",
    "dataset_name": "",
    "date_created" : 12341212,
    "date_modified" : 124123412,
    "versions": {
      "0" : {
        "files" : {
          "some fileid" : {
            "image_version_id" : "0932dn0nw098n3",
            "current_class" : "",
            "original_class" : "",
            "confidence_vectors" : [[]],
            "projections" : {
              "some value": [
                0,
                ...
              ]
            },
            "mean": [0,0,0],
            "std_dev": [0,0,0]
          },
          ...
        },
        "mean": [0,0,0],
        "std_dev": [0,0,0],
      },
      ...
    },
    "reports": {
      "0": {
        "data_id": "901dj012931iu3091",
        "date_created": 12341212,
        "tasks": [
          MynahICProcessTaskType
        ]
      },
      ...
    }
  }
  ```
### `MynahODDataset`
  ```json
  {
    "uuid" : "",
    "owner_uuid": "",
    "dataset_name": "",
    "date_created" : 12341212,
    "date_modified" : 124123412,
    "versions" : {
      "0" : {
        "entities" : {
          "uuid1" : {
            "current_label" : "label1",
            "original_label" : "label2",
            "vertices" : [[3,4], [3,7], [8,7], [8,4]]
          },
          "uuid2" : {
            "current_label" : "label3",
            "original_label" : "label1",
            "vertices" : [[10,4], [7,54], [7,12], [4,6]]
          },
          ...
        },
        "files" : {
          "fileid1" : {
            "image_version_id" : "asidnadasd",
            "entities" : {
              "label1" : [
                "uuid1",
                "uuid2"
              ],
              ...
            }
          },
          "fileid2" : {
            "image_version_id" : "asidnadasd",
            "entities" : {
              "label2" : [
                "uuid1"
              ],
              "label3" : [
                "uuid2"
              ],
              ...
            }
          },
          ...
        },
        "file_entities" : {
          "label1" : [
            "fileid1",
            ...
          ],
          "label2" : [
            "fileid2",
            ...
          ],
          "label3" : [
            "fileid3",
            ...
          ],
          ...
        }
      },
      ...
    }
  }
  ```

### `MynahFile`
  ```json
  {
    "uuid" : "",
    "owner_uuid" : "",
    "name" : "",
    "date_created": 1231312,
    "versions" : {
      "original" : {
        "exists_locally": true,
        "metadata" : {
          "width" : 120,
          "length" : 80,
          "channels" : 3,
          "size": 123123,
          ...
        }
      },
      "latest" : {
         "exists_locally" : true,
         "metadata" : {
           "width" : 120,
           "length" : 80,
           "channels" : 3,
           "size": 123123,
           ... 
        }     
      },
      "dsa98fn983nv9012dqwf4vew" : {
        "exists_locally" : true,
        "metadata" : {
          "width" : 120,
          "length" : 80,
          "channels" : 3,
          "size": 123123,
          ...
        }
      },
      ...
    }
  }
  ```

### `AsyncTaskData`
  ```json
  {
    "started": 141513413,
    "task_id": "awojifd-9023je12",
    "task_status": "pending" | "running" | "completed" | "failed"
  }
  ```

### `MynahICDatasetReport`
  ```json
  {
    "points" : {
      "dog": [
        {
          "fileid" : "<insert id here>",
          "image_version_id" : "<insert id here>",
          "x" : 3.123,
          "y" : 1.112,
          "original_class" : "cat"
        },
        {
          "fileId" : "<insert id here>",
          "image_version_id" : "<insert id here>",
          "x" : 3.123,
          "y" : 1.112,
          "original_class" : "cat"
        }
      ],
      "cat": [
        {
          "fileid" : "<insert id here>",
          "image_version_id" : "<insert id here>",
          "x" : 0,
          "y" : 3,
          "original_class" : "cat"
        },
        {
          "fileId" : "<insert id here>",
          "image_version_id" : "<insert id here>",
          "x" : 0,
          "y" : 3,
          "original_class" : "cat"
        }
      ]
    },
    "tasks": [
      MynahICProcessTaskReport,
      ...
    ]
}
```

### MynahICProcessTaskType
  ```
  "ic::diagnose::mislabeled_images"
  "ic::correct::mislabeled_images"
  "ic::diagnose::class_splitting"
  "ic::correct::class_splitting"
  ```

### MynahICProcessTaskReport
  ```json
  {
    "type": MynahICProcessTaskType,
    "metadata": MynahICProcessTaskReportMetadata
  }
  ```

### MynahICProcessTaskReportMetadata
- One of (by type): 
```
  MynahICProcessTaskDiagnoseMislabeledImagesReport
  MynahICProcessTaskCorrectMislabeledImagesReport
  MynahICProcessTaskDiagnoseClassSplittingReport
  MynahICProcessTaskCorrectClassSplittingReport
```

#### MynahICProcessTaskDiagnoseMislabeledImagesReport


- Type: `"ic::diagnose::mislabeled_images"`
  ```json
  {
    "class_label_errors" : {
      "cat" : {
        "mislabeled" : [ "<fileId1>", "<fileId2>", "..." ],
        "correct" : [ "<fileId3>", "<fileId4>", "..." ]
      },
      "dog" : {
        "mislabeled" : [ "<fileId5>", "<fileId6>", "..." ],
        "correct" : [ "<fileId8>", "<fileId9>", "..." ]
      }
    }
  }
  ```
#### MynahICProcessTaskCorrectMislabeledImagesReport
- Type: `"ic::correct::mislabeled_images"`
  ```json
  {
    "class_label_errors" : {
      "cat" : {
        "mislabeled_corrected" : [ "<fileId1>", "<fileId2>", "..." ],
        "mislabeled_removed" : [ "<fileId5>", "<fileId6>", "..." ],
        "unchanged" : [ "<fileId3>", "<fileId4>", "..." ]
      },
      "dog" : {
        "mislabeled_corrected" : [ "<fileId1>", "<fileId2>", "..." ],
        "mislabeled_removed" : [ "<fileId5>", "<fileId6>", "..." ],
        "unchanged" : [ "<fileId3>", "<fileId4>", "..." ]
      }
    }
  }
  ```
#### MynahICProcessTaskDiagnoseClassSplittingReport
- Type: `"ic::diagnose::class_splitting"`
  ```json
  {
    "classes_splitting" : {
        "cat" : {
            "predicted_classes_count" : 3
        },
        "dog" : {
            "predicted_classes_count" : 2
        }
    }
  }
  ```
#### MynahICProcessTaskCorrectClassSplittingReport
- Type: `"ic::correct::class_splitting"`
  ```json
  {
    "classes_splitting" : {
        "cat" : {
            "new_classes" : [ "cat_split_0", "cat_split_1", "cat_split_2" ]
        },
        "dog" : {
            "new_classes" : [ "dog_split_0", "dog_split_1" ]
        }
    }
  }
  ```


## WebSocket Endpoint
- `GET/POST /api/v1/websocket`

## Upload Endpoint
- `POST /api/v1/upload`
- In the multipart form, `file` must contain the file contents
- Returns the file:
  ```json
  type: MynahFile
  ```
    - `uuid`: The uuid of the file
    - `owner_uuid`: the uuid of the owner
    - `name`: the name of the file
    - `metadata`: file metadata (Note: even though some values are numeric, the structure is a string to string map)

## File endpoints

### Requesting file contents
- All files tracked by Mynah can be requested using:
- `GET /api/v1/file/{uuid}/{tag}`
- Where tag can be `original` or `latest`
- Returns the file contents

### Requesting file metadata
- File metadata can be requested for one or more files
- `GET /api/v1/file/list`
- Params (required)
  - `fileid`
    - Multiple allowed
  - Example: `GET /api/v1/file/list?fileid=123&fileid=234`
- Response:
  ```json
  {
    "123": type: MynahFile,
    "234": type: MynahFile,
     ...
  }
  ```
- Note, providing no `fileid` will return a 4xx status

### Requesting JSON data
- `GET /api/v1/data/json/{id}`
- Example: used to request report data from report metadata `data_id`
- Response: Some json structure

## Admin Management Endpoints

### Creating a User
- `POST /api/v1/admin/user/create`
- Request body
  ```json
  {
    "name_first" : "",
    "name_last" : ""
  }
  ```
    - `name_first`: The first name to assign to the user
    - `name_last`: The last name to assign to the user
- Response body
  ```json
  {
    "jwt" : "",
    "user" : {
      type: MynahUser
    }
  }
  ```
  - `jwt`: The JSON web token that the user must use for authentication
  - `user`: The created user (type `MynahUser`)
    - `uuid`: The unique id of the user
    - `name_first`: The first name to assign to the user
    - `name_last`: The last name to assign to the user

## Dataset GET Endpoints

### List Datasets
- `GET /api/v1/dataset/ic/list`
- Always returns latest versions only
- Response:
  ```json
  [
    type: MynahICDataset
  ]
  ```

- `GET /api/v1/dataset/od/list`
- Always returns latest versions only
- Response:
  ```json
  [
    type: MynahODDataset
  ]
  ```

- `GET /api/v1/dataset/list`
- Always returns latest versions only
- Response:
  ```json
  [
    type: MynahODDataset | MynahICDataset
  ]
  ```

### Get Endpoints
- `GET /api/v1/dataset/ic/{datasetid}`
- Params (optional)
  - `from_version`
    - (Inclusive)
  - `to_version`
    - (Exclusive)
  - Examples: 
    - Requesting the latest version: 
      - `GET /api/v1/dataset/ic/90jj9d20n3d`
    - Requesting a specific version: 
      - `GET /api/v1/dataset/ic/90jj9d20n3d?from_version=0`
    - Requesting the first two versions: 
      - `GET /api/v1/dataset/ic/90jj9d20n3d?from_version=0&to_version=2`
- Response:
  ```json
  [
    type: MynahICDataset
  ]
  ```
- `GET /api/v1/dataset/od/{datasetid}`
- Params (optional)
  - `from_version`
    - (Inclusive)
  - `to_version`
    - (Exclusive)
  - Examples:
    - Requesting the latest version:
      - `GET /api/v1/dataset/od/90jj9d20n3d`
    - Requesting a specific version:
      - `GET /api/v1/dataset/od/90jj9d20n3d?from_version=0`
    - Requesting the first two versions:
      - `GET /api/v1/dataset/od/90jj9d20n3d?from_version=0&to_version=2`
- Response:
  ```json
  [
    type: MynahODDataset
  ]
  ```

## Dataset Action Endpoints

### Creating an image classification dataset
- `POST /api/v1/dataset/ic/create`
- Request body:
  ```json
  {
    "name" : "",
    "files" : {
      "some id" : "some class name",
      ...
    }
  }
  ```
  - `name`: the name to assign the dataset
  - `files` : mapping from fileid to class
- Response body:
  ```json
  type: MynahICDataset
  ```

### Exporting an image classification dataset
- `POST /api/v1/dataset/ic/{datasetid}/export`
- Params (optional)
  - `version`
  - Examples:
    - Requesting the latest version:
      - `GET /api/v1/dataset/ic/90jj9d20n3d/export`
    - Requesting a specific version:
      - `GET /api/v1/dataset/ic/90jj9d20n3d/export?version=4`
- Response is a zip archive

### Starting an image classification process job
- `POST /api/v1/dataset/ic/process/start`
- Request body
  ```json
  {
    "tasks": [
      MynahICProcessTaskType,
      ...
    ],
    "dataset_uuid" : "uuid"
  }
  ```
    - `tasks`: the tasks to run on this dataset (report generated for each)
    - `dataset_uuid`: The id of the dataset to diagnose
- Response:
  ```json
  {
    "task_uuid": "as8fh2n39083nf"
  }
  ```
    - `task_uuid`: Identifier for the async task started (used to query task status, see "Async task endpoints")

## Async task endpoints

### Getting the status of a task
- `GET /api/v1/task/status/{taskid}`
- Response:
  ```json
  AsyncTaskData
  ```

### Listing tasks owned by a user
- `GET /api/v1/task/list`
- Response:
  ```json
  [
    AsyncTaskData
  ]
  ```