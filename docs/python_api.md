# Python/Go Interface

## Functions

### Start Image Classification Model Training Job
- Name: `start_ic_training_job(uuid: str, local_model_save_path : str, request: str, sock_addr: str)`
- `uuid`: The uuid of the user starting the job
- `local_model_save_path`: The local path designating where to save the trained model
- `request`:
  ```json
  {
    "config_params": {
      "model" : "resnet50",
      "min_epochs" : 3,
      "max_epochs" : 50,
      "loss_epsilon" : 0.001,
      "batch_size" : 64,
      "train_test_split" : 0.9,
      "model_save_path" : "<local path>"
    },
    "dataset": "Type::ReiformICDataset (see below)",

  }
  ```
- `sock_addr`: The ipc socket address for sending websocket data
- Output:
```json
  {
    "results" : {
      "model_saved" : true,
      "config_params": {
        "model" : "resnet50",
        "min_epochs" : 3,
        "max_epochs" : 50,
        "loss_epsilon" : 0.001,
        "batch_size" : 64,
        "train_test_split" : 0.9
      },
      "avg_training_loss" : [0.234, 0.221, 0.201, 0.109, 0.091, 0.009, 0.008],
      "all_file_test_loss" : [
        [0.234, 0.221, 0.201, 0.100, 0.091, 0.009, 0.008],
        [0.234, 0.221, 0.201, 0.159, 0.095, 0.009, 0.003],
        [0.234, 0.221, 0.201, 0.119, 0.094, 0.019, 0.04],
        [0.234, 0.221, 0.201, 0.11, 0.092, 0.01, 0.001]
      ],
      "ordered_fileids" : [
        "fileId1",
        "fileId2",
        "fileId3",
        "fileId4"
      ],
      "model_metadata" : {
        "model": "resnet50",
        "image_dims" : [256, 512],
        "type" : "mynah::IC::model",
        "classes" : ["cat", "dog", "jack hay"],
        "mean" : [0.445, 0.544, 0.569],
        "std" : [0.420, 0.069, 0.235],
        "path_to_model" : "local/path/to/saved/model.pt"
      }
    },
    "model_uuid" : "<uuid>"
  }
```

### ReiformICDataset
```json
{
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
    }
    ```

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
