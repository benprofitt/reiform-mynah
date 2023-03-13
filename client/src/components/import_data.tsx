import { Dialog } from "@headlessui/react";
import { useMutation } from "@tanstack/react-query";
import clsx from "clsx";
import { CSSProperties, useRef, useState } from "react";
import { CreateDatasetBody, MynahDataset } from "../types";
import AutoSizer from "react-virtualized-auto-sizer";
import { FixedSizeList as List } from "react-window";
import axios from "axios";
declare module "react" {
  interface InputHTMLAttributes<T> extends HTMLAttributes<T> {
    // extends React's HTMLAttributes
    directory?: string;
    webkitdirectory?: string;
  }
}
export interface ImportDataProps {
  open: boolean;
  close: () => void;
}

interface RowProps {
  index: number;
  style: CSSProperties;
  isFinished: boolean;
  src: string;
  fileName: string;
}

function Row(props: RowProps): JSX.Element {
  const { index, style, fileName, src, isFinished } = props;
  return (
    <div
      className="flex w-full h-[60px] items-center border-b border-grey1"
      style={style}
    >
      <img className="h-[40px] aspect-square mr-[10px]" src={src} />
      {fileName}
      <div
        className={clsx(
          "animate-spin border-t border-b border-l border-sidebarSelected h-[24px] aspect-square rounded-full ml-auto mr-[10px] my-auto",
          isFinished && "hidden"
        )}
      />
    </div>
  );
}

export default function ImportData(props: ImportDataProps): JSX.Element {
  const { open, close } = props;
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [files, setFiles] = useState<{ file: File; isFinished: boolean }[]>();
  const createDatasetMutation = useMutation({
    mutationFn: ({
      dataset,
    }: {
      dataset: CreateDatasetBody;
      _files: File[];
    }) => {
      return fetch("http://localhost:8080/api/v2/dataset/create", {
        method: "POST",
        body: JSON.stringify(dataset),
      });
    },
    onSuccess: async (data, { _files }) => {
      const dataJson = await data.json();
      console.log(dataJson);
      const datasetId: string = dataJson.dataset_id;
      const versionJson = await fetch(
        `http://localhost:8080/api/v2/dataset/${datasetId}/version/refs`
      ).then((res) => res.json());
      const versionId: string = versionJson[0].dataset_version_id;
      _files.forEach((file, ix) =>
        uploadFileMutation.mutate({ file, datasetId, versionId, ix })
      );
    },
  });
  const uploadFileMutation = useMutation({
    mutationFn: async ({
      file,
      datasetId,
      versionId,
    }: {
      file: File;
      datasetId: string;
      versionId: string;
      ix: number;
    }) => {
      const formData = new FormData();
      formData.append("file", file);
      return fetch(
        `http://localhost:8080/api/v2/dataset/${datasetId}/version/${versionId}/upload`,
        {
          method: "POST",
          body: formData,
        }
      );
    },
    onSuccess: (data, { file, ix }) => {
      setFiles((files) => {
        if (files == undefined) return;
        files.splice(ix, 1, { file, isFinished: true });
        return files;
      });
      // i hate making it statey like this because it should just work with the filter but it doesn't...and this fixes it..
      setNumFinished(numFinished => numFinished + 1)
    },
  });
  const [datasetName, setDatasetName] = useState("");
  // const numFinished =
  //   files == undefined
  //     ? 0
  //     : files.filter(({ isFinished }) => isFinished).length;
  const [numFinished, setNumFinished] = useState(0)

  const isValid = Boolean(datasetName);

  const uploadFiles = async (theFiles: File[]) => {
    const _files = theFiles.filter((file) => file.name != ".DS_Store");
    setFiles(
      _files.map((file) => {
        return { file, isFinished: false };
      })
    );
    const dataset: CreateDatasetBody = {
      dataset_name: datasetName,
      dataset_type: "image_classification",
    };
    createDatasetMutation.mutate({ dataset, _files });
  };

  const onClose = () => {
    setFiles(undefined);
    setDatasetName("");
    close();
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
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
            value={datasetName}
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
            directory=""
            webkitdirectory=""
            type="file"
            ref={inputRef}
            onChange={(e) => {
              const uploadedFiles = e.target.files;
              console.log(uploadedFiles);
              if (uploadedFiles === null) return;
              uploadFiles(Array.from(uploadedFiles));
            }}
          />
        </form>
        {files ? (
          <div className="overflow-y-scroll max-h-[500px] w-full">
                <List
                  height={400}
                  width={650}
                  itemCount={files.length}
                  itemSize={40}
                >
                  {({ index, style }) => {
                    const { file, isFinished } = files[index];
                    const { name } = file;
                    const src = URL.createObjectURL(file);
                    return (
                      <Row
                        index={index}
                        style={style}
                        fileName={name}
                        src={src}
                        isFinished={isFinished}
                      ></Row>
                    );
                  }}
                </List>
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
          onClick={onClose}
        >
          Close window
        </button>
        <p
          className={clsx(
            "text-grey2 my-[10px]",
            !files && "opacity-0 select-none"
          )}
        >
          {files !== undefined && numFinished == files.length
            ? "Upload complete"
            : "We will let you know once all of your data sets have been uploaded"}
        </p>
      </div>
    </Dialog>
  );
}
