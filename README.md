# Mynah ML Platform

## Intro
This repository combines the frontend interface and the backend api for Mynah.

![Arch Diagram](docs/mynah_arch_1-13-21.drawio.png)

### Links
- [API Design Document](docs/api_design_doc.md)
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

- To run tests:
```
make test
```

## Setup

### Running the API
- **Important Note**: Running `make clean` will clear the default sqlite database location. If you want to persist your local database, change the name of the database local path in `mynah.json`. The Makefile is set to delete `mynah_local.db`.
- **Important Note**: Running `make clean` will delete the default configuration file `mynah.json`. To persist a configuration file, rename the file and pass the new path using the `-settings` flag (i.e. `./mynah -settings new_name.json`). The Makefile is set to delete `mynah.json`.
- **Important Note**: Running `make clean` will delete the default JWT private key `auth.pem`. To persist a JWT private key, change the name of the PEM file local path in `mynah.json`. The Makefile is set to delete `auth.pem`.

To run the api with a fresh database and configuration file:
```
make clean && make mynah && ./mynah
```

### Using Docker (for frontend development)
- Install Docker on your machine and run
```
docker build -t mynah:latest .
docker run -p 8080:8080 mynah:latest
```

### Using Vagrant (for backend development)
- Install Vagrant on your machine and run
```
vagrant up && vagrant ssh
```
- Once in the VM, run (optionally add to your bash config)
```
export PATH=$PATH:/usr/local/go/bin
export PKG_CONFIG_PATH=/vagrant/python
cd /vagrant
```
