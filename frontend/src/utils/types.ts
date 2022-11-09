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

export interface CreateICDataset {
  name: string;
  files: Record<string, string>;
}
export interface MynahICFile {
  image_version_id: string;
  current_class: string;
  original_class: string;
  confidence_vectors: string[][];
  projections: {
    [value: string]: number[];
  };
  mean: number[];
  std_dev: number[];
}

export interface MynahICVersion {
  files: {
    [fileId: string]: MynahICFile;
  };
  mean: number[];
  std_dev: number[];
}

export interface MynahICReport {
  data_id: string;
  date_created: number;
  tasks: MynahICProcessTaskType[];
}
export interface MynahICDataset {
  uuid: string;
  owner_uuid: string;
  dataset_name: string;
  date_created: number;
  date_modified: number;
  versions: {
    [versionID: string]: MynahICVersion;
  };
  reports: {
    [versionID: string]: MynahICReport;
  };
}

export interface MynahICPoint {
  fileid: string;
  image_version_id: string;
  x: number;
  y: number;
  original_class: string;
}

export type MynahICProcessTaskType =
  | "ic::diagnose::mislabeled_images"
  | "ic::correct::mislabeled_images"
  | "ic::diagnose::class_splitting"
  | "ic::correct::class_splitting";

export interface MynahICProcessTaskDiagnoseMislabeledImagesReport {
  class_label_errors: {
    [className: string]: {
      mislabeled: string[];
      correct: string[];
    };
  };
}

export interface MynahICProcessTaskCorrectMislabeledImagesReport {
  class_label_errors: {
    [className: string]: {
      mislabeled_corrected: string[];
      mislabeled_removed: string[];
      unchanged: string[];
    };
  };
}

export interface MynahICProcessTaskDiagnoseClassSplittingReport {
  classes_splitting: {
    [className: string]: {
      predicted_classes_count: number;
    };
  };
}

export interface MynahICProcessTaskCorrectClassSplittingReport {
  classes_splitting: {
    [className: string]: {
      new_classes: string[];
    };
  };
}

export type MynahICProcessTaskReportMetadata =
  | MynahICProcessTaskDiagnoseMislabeledImagesReport
  | MynahICProcessTaskCorrectMislabeledImagesReport
  | MynahICProcessTaskDiagnoseClassSplittingReport
  | MynahICProcessTaskCorrectClassSplittingReport;

export interface MynahICTaskReport {
  type: MynahICProcessTaskType;
  metadata: MynahICProcessTaskReportMetadata;
}

export interface MynahICDatasetReport {
  points: {
    [className: string]: MynahICPoint[];
  };
  tasks: MynahICTaskReport[];
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

export interface MynahODVersion {
  entities: Record<string, MynahODEntitie>;
  files: Record<string, MynahODFile>;
  file_entities: Record<string, string[]>;
}

export interface MynahODDataset {
  uuid: string;
  owner_uuid: string;
  dataset_name: string;
  date_created: number;
  date_modified: number;
  versions: { [versionID: string]: MynahODVersion };
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

export interface AsyncTaskData {
  started: number;
  task_id: string;
  task_status: "pending" | "running" | "completed" | "failed";
}

export type EitherDataset = MynahICDataset | MynahODDataset;
export type EitherFile = MynahICFile | MynahODFile;
export type EitherVersion = MynahICVersion | MynahODVersion;
