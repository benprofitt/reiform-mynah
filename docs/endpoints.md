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
- Returns the file

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

## Project Action Endpoints
