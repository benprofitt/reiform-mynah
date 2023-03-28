import { useMutation } from "@tanstack/react-query";
import clsx from "clsx";
import { ChangeEvent, CSSProperties, useRef, useState } from "react";
import FileUploadList from "./file_upload_list";

declare module "react" {
  interface InputHTMLAttributes<T> extends HTMLAttributes<T> {
    // extends React's HTMLAttributes
    directory?: string;
    webkitdirectory?: string;
  }
}

export interface FileUploaderProps {
  datasetId: string;
  datasetVersionId: string;
  datasetName: string;
}

export default function FileUploader(props: FileUploaderProps): JSX.Element {
  const { datasetId, datasetVersionId, datasetName } = props;
  const [numFinished, setNumFinished] = useState(0);
  const [files, setFiles] = useState<{ file: File; isFinished: boolean }[]>();
  const inputRef = useRef<HTMLInputElement | null>(null);

  const uploadFiles = (e: ChangeEvent<HTMLInputElement>) => {
    const uploadedFiles = e.target.files;
    if (uploadedFiles === null) return;
    const files = Array.from(uploadedFiles)
      .filter((file) => file.name != ".DS_Store")
      .map((file) => {
        return { file, isFinished: false };
      });
    if (files.length == 0) return;
    setFiles(files);
    files.forEach(({ file }, ix) => uploadFileMutation.mutate({ file, ix }));
  };

  const uploadFileMutation = useMutation({
    mutationFn: async ({ file }: { file: File; ix: number }) => {
      const formData = new FormData();
      formData.append("file", file);
      return fetch(
        `http://localhost:8080/api/v2/dataset/${datasetId}/version/${datasetVersionId}/upload`,
        {
          method: "POST",
          body: formData,
        }
      );
    },
    onSuccess: async (data, { file, ix }) => {
      setFiles((files) => {
        if (files == undefined) return;
        files.splice(ix, 1, { file, isFinished: true });
        return files;
      });
      console.log(ix);
      setNumFinished((numFinished) => numFinished + 1);
      const dataJson = await data.json();
      const fileId = dataJson.file_id;
      const className = file.webkitRelativePath.split("/")[1];
      addClassNameMutation.mutate({ fileId, className });
    },
  });

  const addClassNameMutation = useMutation({
    mutationFn: async ({
      fileId,
      className,
    }: {
      fileId: string;
      className: string;
    }) => {
      const body = {
        assignments: {
          [fileId]: className,
        },
      };
      return fetch(
        `http://localhost:8080/api/v2/dataset/${datasetId}/version/${datasetVersionId}/class`,
        {
          method: "POST",
          body: JSON.stringify(body),
        }
      );
    },
  });
  return (
    <>
      <h1 className="text-[28px] w-full mt-[14px]">
        Upload files to <u>{datasetName}</u>
      </h1>
      <form className="w-full">
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
          onChange={uploadFiles}
        />
        {files == undefined && (
          <button
            type="button"
            className={clsx(
              "text-sm w-full text-left mt-[10px] font-bold mb-[80px] text-blue-500"
            )}
            onClick={() => inputRef.current?.click()}
          >
            Upload from computer
          </button>
        )}
      </form>

      {files != undefined && (
        <>
          <FileUploadList files={files} />
          <p
            className={clsx(
              "text-grey2 my-[10px]",
              !files && "opacity-0 select-none"
            )}
          >
            {numFinished == files.length
              ? "Upload complete"
              : "We will let you know once all of your data sets have been uploaded"}
          </p>
        </>
      )}
    </>
  );
}
