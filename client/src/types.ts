export type MynahDatasetType = "image_classification"

export interface MynahUser {
    user_id: string,
    name_first: string,
    name_last: string,
}

export interface MynahDataset {
    dataset_id: string,
    dataset_name: string,
    date_created: number,
    date_modified: number,
    dataset_type: MynahDatasetType
}

export interface MynahICDatasetVersion {
    dataset_version_id: string,
    version_index: number,
    date_created: number,
    mean: number[],
    std_dev: number[],
    //   task_data: [
    //     TODO
    //   ]
}

export interface MynahICDatasetReport {
    report_id: string,
    date_createad: number,
    created_by: string
}

export interface MynahICDatasetReportContents {
    report_id: string,
    // TODO
}

export interface MynahFile {
    file_id: string,
    name: string,
    date_created: number,
    mime_type: string
}

export interface MynahDatasetVersionRef {
    dataset_version_id: string,
    ancestor_id: string
}

export interface Paginated<T> {
    page: number,
    page_size: number,
    total: number,
    contents: T[]
}

export interface CreateDatasetBody {
    dataset_name: string,
    dataset_type: MynahDatasetType
}