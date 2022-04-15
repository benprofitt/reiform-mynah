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
            }
          },
          ...
        },
        "reports" : {
          "cleaning_report" : "d2d89n923d",
          "diagnosis_report": "493fn39f3f"
        }
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
          "width" : "",
          "length" : "",
          "channels" : "",
          "size": "",
          ...      
        }
      },
      "latest" : {
         "exists_locally" : true,
         "metadata" : {
           "width" : "",
           "length" : "",
           "channels" : "",
           "size": "",
           ...      
        }     
      },
      "dsa98fn983nv9012dqwf4vew" : {
        "exists_locally" : true,
        "metadata" : {
          "width" : "",
          "length" : "",
          "channels" : "",
          "size": "",
          ...
        }
      },
      ...
    }
  }
  ```

### `MynahICDiagnosisReport`
  ```json
  {
    "uuid": "",
    "image_ids" : [
      "fileid1",
      "fileid2",
      ...
    ],
    "image_data": {
      "fileid1": {
        "image_version_id": "908ne9812en128easd2qe12",
        "class": "class1",
        "mislabeled": true,
        "point": {
          "x": 0,
          "y": 0
        },
        "outlier_sets": [
          "lighting"
        ]
      },
      "fileid2": {
        "image_version_id": "asd8932jn9uiqna9sdsar3",
        "class": "class2",
        "mislabeled": false,
        "point": {
          "x": 20,
          "y": 55
        },
        "outlier_sets": []
      },
      ...
    },
    "breakdown": {
      "class1" : {
        "bad": 30,
        "acceptable": 100
      },
      "class2" : {
        "bad": 500,
        "acceptable": 250
      },
      ...
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

## File endpoint
- All files tracked by Mynah can be requested using:
- `GET /api/v1/file/{uuid}/{tag}`
- Where tag can be `original` or `latest`

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
- Response:
  ```json
  [
    type: MynahICDataset
  ]
  ```

- `GET /api/v1/dataset/od/list`
- Response:
  ```json
  [
    type: MynahODDataset
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
    - Requesting a range of versions: 
      - `GET /api/v1/dataset/ic/90jj9d20n3d?from_version=0&to_version=2`
- Response:
  ```json
  [
    type: MynahICDataset
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

### Starting an image classification diagnosis job
- `POST /api/v1/dataset/ic/diagnosis/start`
- Request body
  ```json
  {
    "dataset_uuid" : "uuid"
  }
  ```
    - `dataset_uuid`: The id of the dataset to diagnose

### Getting an image classification diagnosis report
- `GET /api/v1/dataset/ic/report/{reportid}`
- Params
  - `bad_images=true`
  - `class=class1&class=class2 ...`
  - Example: `GET /api/v1/icdataset/report/1?bad_images=true&class=class1&class=class2`
- Response:
  ```json
  type: MynahICDiagnosisReport
  ```
  - Note: when requesting images, use the tag: `report["image_data"]["image_id"]["image_tag"]`