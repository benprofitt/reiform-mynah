# Mynah ML Platform

## Intro
This repository combines the frontend interface and the backend api for Mynah.

### Links
- [API Design Document](docs/api_design_doc.png)
- [API Endpoint Specification](docs/endpoints.md)

### Development
- There are two permanent branches:
  - _Production_
    - This branch should always be stable and working
    - Frozen, LTS versions are periodically finalized from production
    - No direct commits, all merges are code reviewed
  - _Develop_
    - This branch should always be working
    - New features are merged from develop into production once code reviewed and fully tested
    - No direct commits, all development work should be done on a separate branch, squashed and merged in
- In general, all development work should be done on separate feature branches. When features are complete, these branches should be squashed (i.e. commits are combined into one) and merged into develop. Code reviews at this stage would be appropriate.
- When changes to develop are considered stable, new features are merged into production. All changes merged into production _must_ be code reviewed.

### Testing
- We will do our best to maintain test coverage for the api and frontend. In particular, when defects are identified, these should be added as new test cases.
