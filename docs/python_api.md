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
          "/tmp/uuid1.png" : {
            "uuid": "uuid1",
            "width": 32,
            "height": 32,
            "channels": 3,
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03]
          },
          "/tmp/uuid2.png" : {
            "uuid": "uuid2",
            "width": 32,
            "height": 32,
            "channels": 3,
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03]
          }
        },
        "class2" : {
          "/tmp/uuid3.png" : {
            "uuid": "uuid3",
            "width": 32,
            "height": 32,
            "channels": 3,
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03]
          },
          "/tmp/uuid4.jpeg" : {
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
      {"name" : "diagnose_mislabeled_images"},
      {"name" : "correct_lighting_conditions"},
      {"name" : "<diagnose_|correct_>" + "<mislabeled_images|class_splitting|image_blur|lighting_conditions>"}
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
          "/tmp/uuid1.png" : {
            "uuid": "uuid1",
            "current_class": "class2",
            "original_class": "class",
            "width": 32,
            "height": 32,
            "channels": 3,
            "projections": {},
            "confidence_vectors": [[1.0, 2.0]],
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03]
          },
          "/tmp/uuid2.png" : {
            "uuid": "uuid2",
            "current_class": "class2",
            "original_class": "class",
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
          "/tmp/uuid3.png" : {
            "uuid": "uuid3",
            "current_class": "class2",
            "original_class": "class",
            "width": 32,
            "height": 32,
            "channels": 3,
            "projections": {},
            "confidence_vectors": [[1.0, 2.0]],
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03]
          },
          "/tmp/uuid4.jpeg" : {
            "uuid": "uuid4",
            "current_class": "class2",
            "original_class": "class",
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
        "name" : "diagnose_mislabeled_images",
        "outliers" : ["uuid1", "uuid3"]
      },
      {
        "name" : "correct_lighting_conditions",
        "removed" : ["uuid1", "uuid2"],
        "corrected" : ["uuid3"]
      }
    ]
  }
  ```

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
