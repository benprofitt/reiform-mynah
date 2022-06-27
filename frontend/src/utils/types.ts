/*------------------User-------------------------*/
export interface MynahUser {
  uuid: string;
  name_first: string;
  name_last: string;
}

export interface CreateUserResponse {
  jwt: string;
  user: MynahUser;
}

/*--------------ImageClassification-------------*/

export interface MynahICFile {
  [file_id: string]: {
    image_version_id: string;
    current_class: string;
    original_class: string;
    confidence_vectors: string[][];
    projections: {
      [value: string]: number[];
    };
    mean: number[];
    std_dev: number[];
  };
}

export interface CreateICDataset {
  name: string;
  files: {
    [id: string]: string;
  };
}

export interface MynahICData {
  files: MynahICFile;
  mean: number[];
  std_dev: number[];
}

export interface MynahICVersion {
  [version_id: string]: MynahICData
}

export interface MynahICDataset {
  uuid: string;
  owner_uuid: string;
  dataset_name: string;
  date_created: number;
  date_modified: number;
  versions: MynahICVersion
}



/* --------------Object Detection ---------------*/

export interface MynahODFile {
  [file_id: string]: {
    image_version_id: string;
    entities: {
      [label: string]: string[];
    };
  };
}

export interface MynahODEntitie {
  [uuid: string]: {
    current_label: string;
    original_label: string;
    vertices: number[][];
  };
}

export interface MynahODFileEntitie {
  [label: string]: string[];
}

export interface MynahODData {
  entities: MynahODEntitie;
    files: MynahODFile;
    file_entities: MynahODFileEntitie;
}

export interface MynahODVersion {
  [version_id: string]: MynahODData
};

export interface MynahODDataset {
  uuid: string;
  owner_uuid: string;
  dataset_name: string;
  date_created: number;
  date_modified: number;
  versions: MynahODVersion
}

/* ---------------GeneralFiles-------------------*/
export interface MynahFileData {
  exists_locally: boolean;
  metadata: { [id: string]: string };
}

export interface MynahFile {
  uuid: string;
  owner_uuid: string;
  name: string;
  date_created: number;
  versions: {
    original: MynahFileData;
    latest: MynahFileData;
    [id: string]: MynahFileData;
  };
}

export type EitherDataset = MynahICDataset | MynahODDataset
export type EitherFile = MynahICFile | MynahODFile
export type EitherData = MynahICData | MynahODData
export type EitherVersion = MynahICVersion | MynahODVersion
export type DatasetList = EitherDataset[]
