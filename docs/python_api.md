# Python/Go Interface

## Functions

### Start Diagnosis Job
- Name: `start_diagnosis_job(uuid: str, request: str, sock_addr: str)`
- `uuid`: The uuid of the user starting the job
- `request`:
  ```json
  {
    "auto": false,
    "dataset_uuid": "uuid_of_dataset",
    "dataset": {
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
            "uuid": "uuid4.",
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
      {"name" : "mislabeled_images"},
      {"name" : "lighting_conditions"}
    ]
  }
  ```
- `sock_addr`: The ipc socket address for sending websocket data
- Output:
 ```json
  {
    "dataset_uuid": "uuid_of_dataset",
    "tasks" : [
      {
        "name" : "mislabeled_images",
        "datasets": {
          "outliers" : {
            "classes" : ["class1", "class2"],
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03],
            "class_files" : {
              "class1" : {
              },
              "class2" : {
                "fileuuid3" : {
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
                "fileuuid4" : {
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
          "inliers" : {
            "classes" : ["class1", "class2"],
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03],
            "class_files" : {
              "class1" : {
                "fileuuid1" : {
                  "uuid": "uuid1",
                  "current_class": "class1",
                  "original_class": "class",
                  "width": 32,
                  "height": 32,
                  "channels": 3,
                  "projections": {"2d": [1, 2]},
                  "confidence_vectors": [[1.0, 2.0]],
                  "mean": [0.3, 0.4, 0.1],
                  "std_dev": [0.1, 0.12, 0.03]
                },
                "fileuuid2" : {
                  "uuid": "uuid2",
                  "current_class": "class1",
                  "original_class": "class2",
                  "width": 32,
                  "height": 32,
                  "channels": 3,
                  "projections": {"2d": [1, 2]},
                  "confidence_vectors": [[1.0, 2.0]],
                  "mean": [0.3, 0.4, 0.1],
                  "std_dev": [0.1, 0.12, 0.03]
                }
              },
              "class2" : {
              }
            }
          }
        }
      },
      {"name" : "lighting_conditions",
        "datasets": {
          "outliers" : {
            "classes" : ["class1", "class2"],
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03],
            "class_files" : {
              "class1" : {
                "fileuuid1" : {
                  "uuid": "uuid1",
                  "current_class": "class1",
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
                "fileuuid4" : {
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
          "inliers" : {
            "classes" : ["class1", "class2"],
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03],
            "class_files" : {
              "class1" : {
                "fileuuid2" : {
                  "uuid": "uuid2",
                  "current_class": "class1",
                  "original_class": "class2",
                  "width": 32,
                  "height": 32,
                  "channels": 3,
                  "projections": {"2d": [1, 2]},
                  "confidence_vectors": [[1.0, 2.0]],
                  "mean": [0.3, 0.4, 0.1],
                  "std_dev": [0.1, 0.12, 0.03]

                }
              },
              "class2" : {
                "fileuuid3" : {
                  "uuid": "uuid3",
                  "current_class": "class2",
                  "original_class": "class",
                  "width": 32,
                  "height": 32,
                  "channels": 3,
                  "projections": {"2d": [1, 2]},
                  "confidence_vectors": [[1.0, 2.0]],
                  "mean": [0.3, 0.4, 0.1],
                  "std_dev": [0.1, 0.12, 0.03]

                }
              }
            }
          }
        }
      }
    ]
  }
  ```

### Start Cleaning Job
- Name: `start_cleaning_job(uuid: str, request: str, sock_addr: str)`
- `uuid`: The uuid of the user starting the job
- `request`:
  ```json
  {
    "auto": true,
    "dataset_uuid": "uuid_of_dataset",
    "tasks" : [
      {
        "name" : "mislabeled_images",
        "datasets": {
          "outliers" : {
            "classes" : ["class1", "class2"],
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03],
            "class_files" : {
              "class1" : {
              },
              "class2" : {
                "fileuuid3" : {
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
                "fileuuid4" : {
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
          "inliers" : {
            "classes" : ["class1", "class2"],
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03],
            "class_files" : {
              "class1" : {
                "fileuuid1" : {
                  "uuid": "uuid1",
                  "current_class": "class1",
                  "original_class": "class",
                  "width": 32,
                  "height": 32,
                  "channels": 3,
                  "projections": {"2d": [1, 2]},
                  "confidence_vectors": [[1.0, 2.0]],
                  "mean": [0.3, 0.4, 0.1],
                  "std_dev": [0.1, 0.12, 0.03]
                },
                "fileuuid2" : {
                  "uuid": "uuid2",
                  "current_class": "class1",
                  "original_class": "class2",
                  "width": 32,
                  "height": 32,
                  "channels": 3,
                  "projections": {"2d": [1, 2]},
                  "confidence_vectors": [[1.0, 2.0]],
                  "mean": [0.3, 0.4, 0.1],
                  "std_dev": [0.1, 0.12, 0.03]
                }
              },
              "class2" : {
              }
            }
          }
        }
      },
      {"name" : "lighting_conditions",
        "datasets": {
          "outliers" : {
            "classes" : ["class1", "class2"],
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03],
            "class_files" : {
              "class1" : {
                "fileuuid1" : {
                  "uuid": "uuid1",
                  "current_class": "class1",
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
                "fileuuid4" : {
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
          "inliers" : {
            "classes" : ["class1", "class2"],
            "mean": [0.3, 0.4, 0.1],
            "std_dev": [0.1, 0.12, 0.03],
            "class_files" : {
              "class1" : {
                "fileuuid2" : {
                  "uuid": "uuid2",
                  "current_class": "class1",
                  "original_class": "class2",
                  "width": 32,
                  "height": 32,
                  "channels": 3,
                  "projections": {"2d": [1, 2]},
                  "confidence_vectors": [[1.0, 2.0]],
                  "mean": [0.3, 0.4, 0.1],
                  "std_dev": [0.1, 0.12, 0.03]
                }
              },
              "class2" : {
                "fileuuid3" : {
                  "uuid": "uuid3",
                  "current_class": "class2",
                  "original_class": "class",
                  "width": 32,
                  "height": 32,
                  "channels": 3,
                  "projections": {"2d": [1, 2]},
                  "confidence_vectors": [[1.0, 2.0]],
                  "mean": [0.3, 0.4, 0.1],
                  "std_dev": [0.1, 0.12, 0.03]
                }
              }
            }
          }
        }
      }
    ]
  }
  ```
- `sock_addr`: The ipc socket address for sending websocket data
- Output:
  ```json
  {
    "dataset_uuid": "uuid_of_dataset",
      "corrected" : {
        "classes" : ["class1", "class2"],
        "mean": [0.3, 0.4, 0.1],
        "std_dev": [0.1, 0.12, 0.03],
        "class_files" : {
          "class1" : {
          },
          "class2" : {
            "fileuuid3" : {
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
            "fileuuid4" : {
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
        },
      "removed" : {
        "classes" : ["class1", "class2"],
        "mean": [0.3, 0.4, 0.1],
        "std_dev": [0.1, 0.12, 0.03],
        "class_files" : {
          "class1" : {
            "fileuuid1" : {
              "uuid": "uuid1",
              "current_class": "class1",
              "original_class": "class",
              "width": 32,
              "height": 32,
              "channels": 3,
              "projections": {"2d": [1, 2]},
              "confidence_vectors": [[1.0, 2.0]],
              "mean": [0.3, 0.4, 0.1],
              "std_dev": [0.1, 0.12, 0.03]
            },
            "fileuuid2" : {
              "uuid": "uuid2",
              "current_class": "class1",
              "original_class": "class2",
              "width": 32,
              "height": 32,
              "channels": 3,
              "projections": {"2d": [1, 2]},
              "confidence_vectors": [[1.0, 2.0]],
              "mean": [0.3, 0.4, 0.1],
              "std_dev": [0.1, 0.12, 0.03]
            }
          },
          "class2" : {
          }
        }
      }
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
