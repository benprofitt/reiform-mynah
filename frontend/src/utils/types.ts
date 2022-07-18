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
  image_version_id: string;
  current_class: string;
  original_class: string;
  confidence_vectors: string[][];
  projections: Record<string, number[]>;
  mean: number[];
  std_dev: number[];
}

export interface CreateICDataset {
  name: string;
  files: Record<string, string>
}

export interface MynahICData {
  files: Record<string, MynahICFile>;
  mean: number[];
  std_dev: number[];
}

export type MynahICVersion = Record<string, MynahICData>;

export interface MynahICDataset {
  uuid: string;
  owner_uuid: string;
  dataset_name: string;
  date_created: number;
  date_modified: number;
  versions: MynahICVersion;
}

/* --------------Object Detection ---------------*/

export interface MynahODFile {
  image_version_id: string;
  entities: Record<string, string[]>;
}

export interface MynahODEntitie {
  current_label: string;
  original_label: string;
  vertices: number[][];
}

export interface MynahODData {
  entities: Record<string, MynahODEntitie>;
  files: Record<string, MynahODFile>;
  file_entities: Record<string, string[]>;
}

export type MynahODVersion = Record<string, MynahODData>;

export interface MynahODDataset {
  uuid: string;
  owner_uuid: string;
  dataset_name: string;
  date_created: number;
  date_modified: number;
  versions: MynahODVersion;
}

/* ---------------GeneralFiles-------------------*/
export interface MynahFileData {
  exists_locally: boolean;
  metadata: Record<string, string>;
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

export type EitherDataset = MynahICDataset | MynahODDataset;
export type EitherFile = MynahICFile | MynahODFile;
export type EitherData = MynahICData | MynahODData;
export type EitherVersion = MynahICVersion | MynahODVersion;
export type DatasetList = EitherDataset[];
