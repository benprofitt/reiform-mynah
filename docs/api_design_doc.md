# Mynah API Design

## Scope


### Storage of state
- Users
  - Defines permissions on projects
- Projects
  - Defined by data, settings (i.e. repeated transformations on data), users allowed to view/edit project

### Processing of Data
- Workflow
  - User creates project
  - User defines project attributes (i.e. permissions, data format, ML specific settings, processing behaviors)
  - User uploads initial data
    - Produces various outputs based on initial project attributes (including graphs, sample images, etc)
  - User modifies project attributes in feedback loop
  - User is satisfied with attributes, processes remaining data as pipeline
    - In scope:
      - Batch file upload
    - Currently out of scope:
      - Sophisticated upload/scraping (for example, all files in an S3 bucket)
  - Project is saved with final attributes, permissions, etc ("metadata")
    - Currently out of scope:
      - History of changes made to attributes (undo/redo functionality should be considered later)
      - Actual processed data (should be stored by user in their own system)

## Implementation
