# Python/Go Interface

## General response format
```json
{
  "status": 0,
  "data": result | error msg
}
```

## Error codes
- A non-zero status indicates an error:
- Note: on failure, `"data"` should contain the error reason
- `1`: General error

## Functions


### Start Image Correction Processing Job
- Name: `start_ic_processing_job(uuid: str, request: str, sock_addr: str)`
- `uuid`: The uuid of the user starting the job
- `request`:
  ```json
  {
    "config_params": {
      "models_path": "path/to/models/dir/"
    },
    "dataset": {
      "uuid": "uuid_of_dataset",
      "classes" : ["class1", "class2"],
      "mean": [0.3, 0.4, 0.1],
      "std_dev": [0.1, 0.12, 0.03],
      "class_files" : {
        "class1" : {
          "/tmp/uuid1" : {
            "uuid": "uuid1",
            "width": 32,
            "height": 32,
            "channels": 3,
            "projections": {},
            "confidence_vectors": [[1.0, 2.0]],
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03]
          },
          "/tmp/uuid2" : {
            "uuid": "uuid2",
            "width": 32,
            "height": 32,
            "channels": 3,
            "projections": {},
            "confidence_vectors": [[1.0, 2.0]],
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03]
          }
        },
        "class2" : {
          "/tmp/uuid3" : {
            "uuid": "uuid3",
            "width": 32,
            "height": 32,
            "channels": 3,
            "projections": {},
            "confidence_vectors": [[1.0, 2.0]],
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03]
          },
          "/tmp/uuid4" : {
            "uuid": "uuid4",
            "width": 32,
            "height": 32,
            "channels": 3,
            "projections": {},
            "confidence_vectors": [[1.0, 2.0]],
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03]
          }
        }
      }
    },
    "tasks": [ 
      {
        "type" : MynahICProcessTaskType
      }
    ],
    "previous_results" : [ {
        "type" : MynahICProcessTaskType,
        "metadata" : Metadata
      }
    ]
  }
  ```
- `sock_addr`: The ipc socket address for sending websocket data
- Output:
```json
{
  "status": 0,
  "data": {
    "dataset": {
      "uuid": "uuid_of_dataset",
      "classes" : ["class1", "class2"],
      "mean": [0.3, 0.4, 0.1],
      "std_dev": [0.1, 0.12, 0.03],
      "class_files" : {
        "class1" : {
          "/tmp/uuid1" : {
            "uuid": "uuid1",
            "current_class": "class2",
            "projections": {},
            "confidence_vectors": [[1.0, 2.0]],
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03]
          },
          "/tmp/uuid2" : {
            "uuid": "uuid2",
            "current_class": "class2",
            "projections": {},
            "confidence_vectors": [[1.0, 2.0]],
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03]
          }
        },
        "class2" : {
          "/tmp/uuid3" : {
            "uuid": "uuid3",
            "current_class": "class2",
            "projections": {},
            "confidence_vectors": [[1.0, 2.0]],
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03]
          },
          "/tmp/uuid4" : {
            "uuid": "uuid4",
            "current_class": "class2",
            "projections": {},
            "confidence_vectors": [[1.0, 2.0]],
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03]
          }
        }
      }
    },
    "tasks": [
      {
        "type" : MynahICProcessTaskType,
        "metadata": Metadata
      },
      {
        "type" : MynahICProcessTaskType,
        "metadata": Metadata
      }
    ]
  }
}
  
  ```

#### MynahICProcessTaskType
```
  "ic::diagnose::mislabeled_images"
  "ic::correct::mislabeled_images"
  "ic::diagnose::class_splitting"
  "ic::correct::class_splitting"
```

#### Metadata
- `"ic::diagnose::mislabeled_images"`
  ```json
  {
    "outliers": ["fileid", ...]
  }
  ```
- `"ic::correct::mislabeled_images"`
  ```json
  {
    "removed": ["fileid", ...],
    "corrected": ["fileid", ...]
  }
  ```  
- `"ic::diagnose::class_splitting"`
  ```json
  {
    "predicted_class_splits" : {
      "class1" : [["fileid1", ...], ["fileid4", ...], ["fileid420", ...], ...],
      "class2" : [["fileid42", ...], ["fileid421", ...], ...]
    }
  }
  ```  
- `"ic::correct::class_splitting"`
  ```json
  {
    "actual_class_splits" : {
      "class1" : {"class1" : ["fileid1", ...], "class1_split_1" : ["fileid4", ...], "class1_split_2" : ["fileid420", ...], ...},
      "class2" : {"class2" : ["fileid42", ...], "class2_split_1" : ["fileid421", ...], ...}
    }
  }
  ```

### Get Image Metadata
- Name: `get_metadata_for_images(uuid: str, request: str, sock_addr: str)`
- `uuid`: The uuid of the user starting the job
- `request` :
  ```json
  {
    "images": [
      {
        "uuid": "some_image_id",
        "path": "/path/to/file"
      },
      ...
    ]
  }
  ```
- `sock_addr`: The ipc socket address for sending websocket data
- Output:
```json
{
  "status": 0,
  "data": {
    "images": {
      "some_image_id": {
        "channels": 3,
        "height": 32,
        "width": 64,
        "mean": [
          0.3,
          0.4,
          0.1
        ],
        "std_dev": [
          0.1,
          0.12,
          0.03
        ]
      },
      ...
    }
  }
}
```
