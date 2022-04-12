# Mynah CLI Tool

```shell
Usage of ./mynah-cli:
  -address string
        server address (default "127.0.0.1:8080")
  -jwt string
        auth jwt
  -proto string
        http or https (default "http")
  -task string
        task path (default "task.json")
```

## Task Format

- `tasks` list: array of `task` objects
- `task` object:
  - `task_id`: user-defined identifier for referencing previous steps
  - `task_type`: the type of the task
  - `task_data`: task specific data (depends on `task_type`)

### Upload task
- `file_paths`: paths of files to upload
- `multithread`: whether to upload files in parallel

## Examples
- Uploading files
```json
{
  "tasks" : [
    {
      "task_id": "upload_files",
      "task_type": "mynah::upload",
      "task_data": {
        "file_paths" : [
          "docs/test_image.jpg",
          "docs/test_image.jpg",
          "docs/test_image.jpg",
          "docs/test_image.jpg",
          "docs/test_image.jpg"
        ],
        "multithread" : true
      }
    }
  ]
}
```