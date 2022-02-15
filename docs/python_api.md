# Python/Go Interface

## Functions

### Start Diagnosis Job
- Name: `start_diagnosis_job(uuid: str, request: str, sock_addr: str)`
- `uuid`: The uuid of the user starting the job
- `request`:
  ```json
  {
    "classes" : ["class1", "class2"],
    "files" : {
      "class1" : {
        "fileuuid1" : {
          "uuid": "uuid1",
          "current_class": "class1",
          "original_class": "class",
          "projections": {},
          "confidence_vectors": [[1.0, 2.0]],
          "mime_type" : "image/tiff",
        },
        "fileuuid2" : {
          "uuid": "uuid2",
          "current_class": "class1",
          "original_class": "class2",
          "projections": {},
          "confidence_vectors": [[1.0, 2.0]],
          "mime_type" : "image/png",
        }
      },
      "class2" : {
        "fileuuid3" : {
          "uuid": "uuid3",
          "current_class": "class2",
          "original_class": "class",
          "projections": {},
          "confidence_vectors": [[1.0, 2.0]],
          "mime_type" : "image/jpeg",
        },
        "fileuuid4" : {
          "uuid": "uuid4",
          "current_class": "class2",
          "original_class": "class",
          "projections": {},
          "confidence_vectors": [[1.0, 2.0]],
          "mime_type" : "image/jpeg",
        }
      }
    }
  }
  ```
- `sock_addr`: The ipc socket address for sending websocket data
- Output:
  ```json

  ```
