import { Dialog } from "@headlessui/react";
import { useMutation } from "@tanstack/react-query";
import clsx from "clsx";
import { useRef, useState } from "react";
import { CreateDatasetBody, MynahDataset } from "../types";

export interface ImportDataProps {
  open: boolean;
  close: () => void;
}

export default function ImportData(props: ImportDataProps): JSX.Element {
  const { open, close } = props;
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dataset, setDataset] = useState<MynahDataset>()
  const createDatasetMutation = useMutation({
    mutationFn: (dataset: CreateDatasetBody) => {
      return fetch('/api/v2/dataset/create', {
        method: "POST",
        body: JSON.stringify(dataset)
      }).then(res => res.json()).then((resJson) => {
        if (resJson == null) return
        setDataset(resJson)
        console.log(resJson)
      })
    }
  })
  const uploadFileMutation = useMutation({
    mutationFn: (file: File) => {
      return fetch(`/api/v2/dataset/${dataset?.dataset_id}/version/0/upload`, {
        method: "POST"
      })
    }
  })
  const [datasetName, setDatasetName] = useState("");
  const [files, setFiles] = useState<File[]>()
  const [numFinished, setNumFinished] = useState(0);

  const isValid = Boolean(datasetName);

  const uploadFiles = async (theFiles: File[]) => {
    setFiles(
      theFiles.filter((file) => file.name != '.DS_Store')
    );
    const dataset: CreateDatasetBody = {
      dataset_name: datasetName,
      dataset_type: "image_classification"
    };
    createDatasetMutation.mutate(dataset)
  };

  return (
    <Dialog
      open={open}
      onClose={close}
      className="fixed inset-0 z-20 /w-full /h-full flex items-center justify-center py-[10px]"
    >
      <Dialog.Overlay className="w-full h-full bg-black absolute  top-0 left-0 opacity-20 z-20" />
      <div className="w-[752px] h-fit max-h-full mx-auto flex flex-col items-center relative z-30 bg-white px-[24px]">
        <h1 className="text-[28px] w-full mt-[14px]">Create new data set</h1>
        <form className="w-full">
          <h3 className="font-black">Data set name</h3>
          <input
            className="w-full border border-grey1 focus:outline-none focus:ring-0 h-[56px] mt-[10px] pl-[10px]"
            placeholder="Name collection"
            onChange={(e) => setDatasetName(e.target.value)}
          />
          <div className="font-black w-full border-b border-grey1 pb-[10px] mt-[30px] flex justify-between">
            <h3>Upload Files</h3>
            {files && (
              <p className="font-medium">
                {numFinished} of {files.length} images uploaded
              </p>
            )}
          </div>
          <input
            className="hidden"
            id="upload"
            type="file"
            ref={inputRef}
            // directory=""
            // webkitdirectory=""
            onChange={(e) => {
              const uploadedFiles = e.target.files;
              if (uploadedFiles === null) return;
              uploadFiles(Array.from(uploadedFiles));
            }}
          />
        </form>
        {files ? (
          <div className="overflow-y-scroll max-h-[500px] w-full">
            {files.map((file, ix) => {
              const filename = file.name;
              const src = URL.createObjectURL(file);
              return (
                <div
                  className="flex w-full h-[60px] items-center border-b border-grey1"
                  key={ix}
                >
                  <img className="h-[40px] aspect-square mr-[10px]" src={src} />
                  {filename}
                  <div
                    className={clsx(
                      "animate-spin border-t border-b border-l border-sidebarSelected h-[24px] aspect-square rounded-full ml-auto mr-[10px] my-auto",
                      // isFinished && "hidden"
                    )}
                  />
                </div>
              );
            })}
          </div>
        ) : (
          <button
            type="button"
            className={clsx(
              "text-sm w-full text-left mt-[10px] font-bold mb-[80px]",
              isValid ? "text-blue-500 hover:text-blue-700" : "text-grey2"
            )}
            disabled={!isValid}
            onClick={() => inputRef.current?.click()}
          >
            Upload from computer
          </button>
        )}
        <button
          className={clsx(
            "w-full text-white h-[40px] font-bold text-[16px] shrink-0",
            files ? "bg-blue-500" : "bg-grey1"
          )}
          onClick={close}
        >
          Close window
        </button>
        {/* <p
          className={clsx(
            "text-grey2 my-[10px]",
            !files && "opacity-0 select-none"
          )}
        >
          {isUploadFinished
            ? "Upload complete"
            : "We will let you know once all of your data sets have been uploaded"}
        </p> */}
      </div>
    </Dialog>
  );
}
