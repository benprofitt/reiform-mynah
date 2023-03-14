import { useMutation } from "@tanstack/react-query";
import clsx from "clsx";
import { ChangeEvent, CSSProperties, useRef, useState } from "react";
import AutoSizer from "react-virtualized-auto-sizer";
import { FixedSizeList as List } from "react-window";

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

interface FilePreviewProps {
  style: CSSProperties;
  isFinished: boolean;
  src: string;
  fileName: string;
}

function FilePreview(props: FilePreviewProps): JSX.Element {
  const { style, fileName, src, isFinished } = props;
  return (
    <li
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
    </li>
  );
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
    if (files.length == 0) return
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
      console.log(ix)
      setNumFinished((numFinished) => numFinished + 1);
      const dataJson = await data.json()
      const fileId = dataJson.file_id
      const className = file.webkitRelativePath.split("/")[1]
      addClassNameMutation.mutate({fileId, className})
    },
  });

  const addClassNameMutation = useMutation({
    mutationFn: async ({ fileId, className } : { fileId: string; className: string }) => {
        const body = {
            assignments: {
                [fileId]: className
            }
        }
        return fetch(`http://localhost:8080/api/v2/dataset/${datasetId}/version/${datasetVersionId}/file/classes`, {
            method: "POST",
            body: JSON.stringify(body)
        })
    }})
  return (
    <>
      <h1 className="text-[28px] w-full mt-[14px]">
        Upload files to {datasetName}
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
              "text-sm w-full text-left mt-[10px] font-bold mb-[80px]"
            )}
            onClick={() => inputRef.current?.click()}
          >
            Upload from computer
          </button>
        )}
      </form>

      {files != undefined && (
        <ul className="h-[500px] w-full">
          <AutoSizer>
            {({ height, width }) => (
              <List
                className="no-scrollbar"
                height={height}
                width={width}
                itemCount={files.length}
                itemSize={60}
              >
                {({ index, style }) => {
                  const { file, isFinished } = files[index];
                  const { name } = file;
                  const src = URL.createObjectURL(file);
                  return (
                    <FilePreview
                      style={style}
                      fileName={name}
                      src={src}
                      isFinished={isFinished}
                    />
                  );
                }}
              </List>
            )}
          </AutoSizer>
        </ul>
      )}
    </>
  );
}
