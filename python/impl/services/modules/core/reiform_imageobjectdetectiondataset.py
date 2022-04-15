from __future__ import annotations
from .reiform_imagedataset import *

class ReiformIODDataset(ReiformImageDataset):
    def __init__(self, classes: List[str] = []) -> None:
        super().__init__(classes)

        # Maps file_uuid to file
        self.files : Dict[str, ReiformIODFile] = {}

        # Maps entity_uuid to entity
        self.entities : Dict[str, ReiformIODEntity] = {}

        # Maps classname to list of files containing that class
        self.file_entities : Dict[str, List[str]] = {}

        for c in self.class_list:
            self.file_entities[c] = []

class ReiformIODFile(ReiformImageFile):
    def __init__(self, name: str) -> None:
        super().__init__(name)

        # Map from classname to the uuids for all objects of that class
        self.entities : Dict[str, str] = {}

class ReiformIODEntity(ReiformImageEntity):
    def __init__(self) -> None:
        super().__init__()
        
        self.vertices : List[Tuple[int, int]] = []
