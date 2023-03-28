import { Dialog, Tab } from "@headlessui/react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { FormEvent, useState } from "react";
import {
  CreateDatasetBody,
  MynahDataset,
  MynahDatasetVersionRef,
} from "../types";
import FileUploader from "./file_uploader";
import { ReactComponent as XOut } from "../assets/XOut.svg";
import makeRequest from "../utils/apiFetch";

export interface ImportDataProps {
  open: boolean;
  close: () => void;
}

export default function ImportData(props: ImportDataProps): JSX.Element {
  const { open, close } = props;
  const queryClient = useQueryClient();
  const [beganCreation, setBeganCreation] = useState(false);
  const [datasetName, setDatasetName] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [datasetVersionId, setDatasetVersionId] = useState("");
  const readyToCreateDataset = Boolean(datasetName);
  const readyToUploadFiles = Boolean(datasetId) && Boolean(datasetVersionId);
  const createDatasetMutation = useMutation({
    mutationFn: (dataset: CreateDatasetBody) => {
      return makeRequest<MynahDataset>(
        "POST",
        "/api/v2/dataset/create",
        JSON.stringify(dataset)
      );
    },
    onSuccess: async (data) => {
      const { dataset_id } = data;
      setDatasetId(dataset_id);
      const [versionData, ...rest] = await makeRequest<
        MynahDatasetVersionRef[]
      >("GET", `/api/v2/dataset/${dataset_id}/version/refs`);
      setDatasetVersionId(versionData.dataset_version_id);
      queryClient.refetchQueries({ queryKey: ["datasets"] });
    },
  });

  const createDataset = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!readyToCreateDataset) return;
    setBeganCreation(true);
    const dataset: CreateDatasetBody = {
      dataset_name: datasetName,
      dataset_type: "image_classification",
    };
    createDatasetMutation.mutate(dataset);
  };

  const onClose = () => {
    setBeganCreation(false);
    setDatasetName("");
    setDatasetVersionId("");
    setDatasetId("");
    close();
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      className="fixed inset-0 z-20 /w-full /h-full flex items-center justify-center"
    >
      <Dialog.Overlay className="w-full h-full bg-black absolute  top-0 left-0 opacity-20 z-20" />
      <div className="w-[752px] h-fit max-h-full mx-auto flex flex-col items-center relative z-30 bg-white px-[24px] py-[20px]">
        <button className="absolute right-[5px] top-[5px]" onClick={onClose}>
          <XOut />
        </button>

        {!readyToUploadFiles ? (
          <>
            <h1 className="text-[28px] w-full">Create new data set</h1>
            <form className="w-full" onSubmit={createDataset}>
              <h3 className="font-black">Data set name</h3>
              <input
                disabled={beganCreation}
                className="w-full border border-grey1 focus:outline-none focus:ring-0 h-[56px] mt-[10px] pl-[10px]"
                placeholder="Name collection"
                value={datasetName}
                onChange={(e) => setDatasetName(e.target.value)}
              />
              <button
                type="submit"
                disabled={!readyToCreateDataset || beganCreation}
                className={clsx(
                  "w-full text-white h-[40px] font-bold text-[16px] flex justify-center items-center mt-[10px]",
                  readyToCreateDataset ? "bg-blue-500" : "bg-gray-300"
                )}
              >
                Create data set and proceed to file upload
                {beganCreation && !readyToUploadFiles && (
                  <div className="animate-spin h-[20px] aspect-square border-r-white border-2 rounded-full mr-[10px]" />
                )}
              </button>
            </form>
          </>
        ) : (
          <FileUploader
            datasetId={datasetId}
            datasetName={datasetName}
            datasetVersionId={datasetVersionId}
          />
        )}
      </div>
    </Dialog>
  );
}
