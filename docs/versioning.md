# Data versioning

## Dataset Versioning
- Datasets contain references to files and information about the files.
- Data is versioned using tags incrementing tags.
    - When operations modify a dataset, a new version is created using the operation results. This is given a new tag.
        - This process will create a new version of every file in the dataset for the previous dataset version, if necessary, to retain consistency between file metadata and the raw file.
            - Because file versions are keyed by hash, if the file is identical to a previous version, it will not be copied more than once (the hash will already exist).
            - If the file hash does not exist, the `latest` version will be duplicated and tagged with the hash of the file. This allows further modification to the `latest` file version.
        - The latest version of the dataset will use the latest versions of the files it contains
- Reports generated for versions of a dataset will reference versions of images

Note: the implication here is that dataset versions and file versions are immutable to maintain this link.

### ICDataset
- `Files` data is versioned by "tag"

### ODDataset
- `Entities`, `Files`, and `FileEntities` are versioned by "tag"

## File Versioning
- Files contain information about uploaded files.
- Files are versioned using tags. Two tags, `original` and `latest` are populated by default.
    - Changes are only made to the `latest` version.
    - New tags are created through dataset versioning, by taking the `SHA1` of the file contents.
    - Only the latest version of a file should be modified