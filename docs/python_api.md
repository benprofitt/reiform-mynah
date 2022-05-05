# Python/Go Interface

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
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03]
          },
          "/tmp/uuid2" : {
            "uuid": "uuid2",
            "width": 32,
            "height": 32,
            "channels": 3,
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
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03]
          },
          "/tmp/uuid4" : {
            "uuid": "uuid4",
            "width": 32,
            "height": 32,
            "channels": 3,
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03]
          }
        }
      }
    },
    "tasks": [ 
      {"type" : MynahICProcessTaskType}
    ]
  }
  ```
- `sock_addr`: The ipc socket address for sending websocket data
- Output:
```json
  {
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
  ```

#### MynahICProcessTaskType
```
  "ic::diagnose::mislabeled_images"
  "ic::correct::mislabeled_images"
  "ic::diagnose::class_splitting"
  "ic::correct::class_splitting"
  "ic::diagnose::lighting_conditions"
  "ic::correct::lighting_conditions"
  "ic::diagnose::image_blur"
  "ic::correct::image_blur"
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
  
  }
  ```  
- `"ic::diagnose::class_splitting"`
  ```json
  {
  
  }
  ```  
- `"ic::correct::class_splitting"`
  ```json
  {
  
  }
  ```  
- `"ic::diagnose::lighting_conditions"`
  ```json
  {
  
  }
  ```  
- `"ic::correct::lighting_conditions"`
  ```json
  {
    "removed": ["fileid", ...],
    "corrected": ["fileid", ...]
  }
  ```  
- `"ic::diagnose::image_blur"`
  ```json
  {
  
  }
  ```  
- `"ic::correct::image_blur"`
  

### Get Image Metadata
- Name: `get_image_metadata(uuid: str, request: str, sock_addr: str)`
- `uuid`: The uuid of the user starting the job
- `request` :
  ```json
  {
    "path" : ""
  }
  ```
- `sock_addr`: The ipc socket address for sending websocket data
- Output:
  ```json
    {
      "channels" : 3,
      "height" : 32,
      "width" : 64,
      "mean": [0.3, 0.4, 0.1],
      "std_dev": [0.1, 0.12, 0.03]
    }
  ```
