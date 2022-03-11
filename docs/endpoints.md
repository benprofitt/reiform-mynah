# API Endpoint Specification

## Repeated Types
- `MynahUser`
  ```json
  {
    "uuid" : "",
    "name_first" : "",
    "name_last" : ""
  }
  ```

## GraphQL Endpoints
- `GET/POST /api/v1/graphql/project`
- `GET/POST /api/v1/graphql/user`

## WebSocket Endpoint
- `GET/POST /api/v1/websocket`

## Upload Endpoint
- `POST /api/v1/upload`
- In the multipart form, `file` must contain the file contents
- Returns the file:
  ```json
  {
    "uuid" : "",
    "owner_uuid" : "",
    "name" : "",
    "metadata" : {
      "width" : "",
      "length" : "",
      "channels" : "",
      ...
    }
  }
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
      "uuid" : "",
      "name_first" : "",
      "name_last" : ""
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
  {
    "dataset" : {
      "uuid" : "",
      "owner_uuid": "",
      "dataset_name": "",
      "files" : {
        "some fileid" : {
          "current_class" : "",
          "original_class" : "",
          "confidence_vectors" : [[]]
        }     
      }  
    }
  }
  ```
- `dataset`: the created dataset

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
  ```json
  {
    "project" : {
      "uuid" : "",
      "user_permissions" : {
        "some uuid" : 0
      },
      "project_name" : "",
      "datasets" : [
        "dataset id"
      ]    
    }
  }
  ```
- `project`: the created project

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
