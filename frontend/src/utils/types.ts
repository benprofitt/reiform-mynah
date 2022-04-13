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
  [id: string]: {
    current_class: string;
    original_class: string;
    confidence_vectors: string[][];
  };
}

export interface CreateICDataset {
  name: string;
  files: {
    [id: string]: string;
  };
}

export interface MynahICDataset {
  uuid: string;
  owner_uuid: string;
  dataset_name: string;
  files: MynahICFile[];
}

export interface CreateICProject {
  name: string;
  datasets: string[];
}

export interface MynahICProject {
  uuid: string;
  user_permissions: {
    [id: string]: number;
  };
  project_name: string;
  reports: string[];
  datasets: string[];
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
  versions: {
    original: MynahFileData;
    latest: MynahFileData;
  };
}
