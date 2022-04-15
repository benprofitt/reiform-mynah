# Mynah API Design

## Scope

### Authentication
Because we are offering a non-managed service to users, we should not make the assumption that all users will set up the service using https encryption. Therefore, prompting users for a password would create a security risk since users often reuse their password. Instead, we should use JWT authentication in an HTTP header. The JWT should be signed using a user-controlled (or pregenerated) key in a pem file.
- In scope:
  - Allow admins to create new users (amounts to generating a new JWT and creating a link that new users can navigate to)
  - JWT storage in cookies
- Currently out of scope:
  - JWT rotation
  - Password authentication

When the application is first started, an admin user is created. Token recovery can only be performed by the admin.

For the managed service, JWTs will be managed/rotated by Cognito/OAuth etc

### Storage of state
- Users
  - Unique identifiers and optional identifying information (i.e. name)
- Datasets
  - Data settings
  - Users with permissions for dataset (including the owner)

### Processing of Data
- Workflow
  - User creates dataset
  - User defines dataset attributes (i.e. permissions, data format, ML specific settings, processing behaviors)
  - User uploads initial data
    - Produces various outputs based on initial dataset attributes (including graphs, sample images, etc)
  - User modifies dataset attributes in feedback loop
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
