import clsx from "clsx";
import { useState } from "react";
import makeRequest from "../utils/apiFetch";
import { CreateICDataset, MynahFile, MynahICDataset } from "../utils/types";

export interface ImportDataProps {
  setDatasets: React.Dispatch<React.SetStateAction<MynahICDataset[]>>;
  setSelectedDatasets: React.Dispatch<React.SetStateAction<string[]>>;
}

export default function ImportData(props: ImportDataProps): JSX.Element {
  const { setDatasets, setSelectedDatasets } = props;
  const [thisDevice, setThisDevice] = useState(false);
  const [datasetName, setDatasetName] = useState("");
  const [totalFiles, setTotalFiles] = useState(0);
  const [filesUploaded, setFilesUploaded] = useState(0);

  const uploadFiles = async (files: File[]) => {
    const total = files.length;
    if (total === 0) return;
    setTotalFiles(total);
    const dataset: CreateICDataset = {
      name: datasetName,
      files: {},
    };
    for (let i = 0; i < total; i++) {
      const file = files[i];
      const formData = new FormData();
      const res = await makeRequest<MynahFile>(
        "POST",
        formData,
        "/api/v1/upload",
        "multipart/form-data"
      ).catch((err) => {
        alert("something went wrong with file upload");
        console.log(err);
      });
      if (!res) return;
      dataset.files[res.uuid] = file.webkitRelativePath.split("/")[1];
      setFilesUploaded(i + 1);
    }
    await makeRequest<MynahICDataset>(
      "POST",
      JSON.stringify(dataset),
      "/api/v1/icdataset/create",
      "application/json"
    )
      .then((res) => {
        setDatasets((datasets) => [res, ...datasets]);
        setSelectedDatasets((selectedDatasets) => [
          res.uuid,
          ...selectedDatasets,
        ]);
        setThisDevice(false);
        setDatasetName("");
        setTotalFiles(0);
        setFilesUploaded(0);
      })
      .catch((err) => {
        alert("error making dataset");
        console.log(err);
      });
  };

  return (
    <div className="mt-4 w-96 h-64 border-2 border-black flex flex-col items-center relative">
      <h3 className="h-8 text-center border-b-2 border-black w-full">
        New Dataset
      </h3>
      {totalFiles > 0 && (
        <div className="absolute top-8 right-3">
          {filesUploaded}/{totalFiles}
        </div>
      )}
      <div className="mt-4 w-3/4 border-2 border-black text-center">Source</div>
      <button
        className="mt-4 w-1/2 border-2 border-black text-center"
        onClick={() => setThisDevice(!thisDevice)}
      >
        This device
      </button>
      <div
        className={clsx("flex flex-col w-full px-4", !thisDevice && "hidden")}
      >
        {/* below is the actual file upload component but its ugly so we just trigger it with a button instead */}
        <input
          className="hidden"
          id="upload"
          type="file"
          directory=""
          webkitdirectory=""
          onChange={(e) => {
            const uploadedFiles = e.target.files;
            if (uploadedFiles === null) return;
            uploadFiles(Array.from(uploadedFiles));
          }}
        />
        <label className="flex flex-col justify-start">
          First - Enter dataset name
          <input
            type="text"
            value={datasetName}
            placeholder="Dataset name"
            className="border border-black"
            onChange={(e) => setDatasetName(e.target.value)}
          />
        </label>
        <label
          className={clsx(
            "flex flex-col justify-center",
            datasetName.length < 3 && "hidden"
          )}
        >
          Second - Choose files. <br />
          The upload begins automatically.
          <button
            className="text-white text-sm rounded px-2 h-8 bg-blue-500 hover:bg-blue-700"
            onClick={() => document.getElementById("upload")?.click()}
          >
            Select the directory with a subdirectory for each class
          </button>
        </label>
      </div>
    </div>
  );
}
