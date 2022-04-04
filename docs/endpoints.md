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
    "files" : {
      "some fileid" : {
        "current_class" : "",
        "original_class" : "",
        "confidence_vectors" : [[]]
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
        "entities" : {
          "label1" : [
            "uuid1",
            "uuid2"
          ],
          ...
        }
      },
      "fileid2" : {
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
  }
  ```
### `MynahICProject`
  ```json
  {
    "uuid" : "",
    "user_permissions" : {
      "some uuid" : 0,
      ...
    },
    "project_name" : "",
    "datasets" : [
      "ic dataset id",
      ...
    ],
    "reports" : [
      "reportid1",
      "reportid2",
      ...
    ]
  }
  ```

### `MynahODProject`
  ```json
  {
    "uuid" : "",
    "user_permissions" : {
      "some uuid" : 0,
      ...
    },
    "project_name" : "",
    "datasets" : [
      "od dataset id",
      ...
    ]
  }
  ```

### `MynahFile`
  ```json
  {
    "uuid" : "",
    "owner_uuid" : "",
    "name" : "",
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
      } 
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

## Dataset Action Endpoints

### Creating an image classification dataset
- `POST /api/v1/icdataset/create`
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

## Project Action Endpoints

### Creating an image classification project
- `POST /api/v1/icproject/create`
- Request body:
  ```json
  {
    "name" : "",
    "datasets" : [
      "some id"
    ]
  }
  ```
  - `name`: the name to assign the project
  - `datasets` : datasets by id to include:
- Response:
  ```json
  type: MynahICProject
  ```

### Creating an object detection project
- `POST /api/v1/odproject/create`
- Request body:
  ```json
  {
    "name" : "",
    "datasets" : [
      "some id"
    ]
  }
  ```
  - `name`: the name to assign the project
  - `datasets` : datasets by id to include:
- Response:
  ```json
  type: MynahODProject
  ```

### Starting an image classification diagnosis job
- `POST /api/v1/ic/diagnosis/start`
- Request body
  ```json
  {
    "project_uuid" : "uuid"
  }
  ```
    - `project_uuid`: The id of the project to diagnose
      - Combines datasets in the project and analyzes all files

### Getting an image classification diagnosis report
- `GET /api/v1/icproject/report/{reportid}`
- Params
  - `bad_images=true`
  - `class=class1&class=class2 ...`
  - Example: `GET /api/v1/icproject/report/1?bad_images=true&class=class1&class=class2`
- Response:
  ```json
  type: MynahICDiagnosisReport
  ```