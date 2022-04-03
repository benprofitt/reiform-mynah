# Extensions

Mynah will support various external services for storage/compute/database/auth etc

Extensions are enabled/disabled using go build tags and configured using the Mynah settings file

## Replacement Extensions
Certain services can be used to _replace_ local services. Authentication and database are the two primary candidates. 
In these situations, an external service will be used _instead_ of the local version. 

## Supplementary Extensions
Certain services can be used to _supplement_ local services. For example, exporting files to or importing files from 
AWS S3 does not replace the need to have a local file storage management system. Additionally, it may be desireable to 
support _multiple_ alternate file export destinations.

### Enabling Extensions
Extensions are included in a binary explicitly using go build tags. For example, to include support for s3:
```shell
make mynah GO_ARGS="-tags s3"
```