import { Dialog, Listbox } from "@headlessui/react";
import clsx from "clsx";
import { useState } from "react";
import makeRequest from "../utils/apiFetch";
import { CreateICDataset, MynahFile, MynahICDataset } from "../utils/types";
import ArrowIcon from "../images/ArrowIcon.svg";
import { useQuery } from "react-query";

export interface ImportDataProps {
  open: boolean;
  close: () => void;
  refetch?: () => void;
  // setDatasets: React.Dispatch<React.SetStateAction<MynahICDataset[]>>;
  // setSelectedDatasets: React.Dispatch<React.SetStateAction<string[]>>;
}

const options = [
  { id: 1, name: "Image Classification" },
  { id: 2, name: "Object Detection" },
];

export default function ImportData(props: ImportDataProps): JSX.Element {
  const { refetch } = useQuery("datasets", () =>
  makeRequest("GET", "/api/v1/dataset/list"));
  const { open, close } = props;
  const [datasetName, setDatasetName] = useState("");
  const [selectedType, setSelectedType] = useState<{
    id: Number;
    name: string;
  }>();
  const isValid = Boolean(datasetName && selectedType);
  const [files, setFiles] = useState<{ file: File; isFinished: boolean }[]>();
  // const numUploaded = files?.filter(({ isFinished }) => isFinished).length;
  const [numFinished, setNumFinished] = useState(0);
  // it seems like this state-y numFinished is necessary for
  // getting rerenders to fire on individual upload completions
  // (otherwise it just rerenders once every file is finished...)
  const isUploadFinished = files && numFinished === files.length;

  const uploadFiles = async (theFiles: File[]) => {
    setFiles(
      theFiles.map((file) => {
        return { file, isFinished: false };
      })
    );
    const dataset: CreateICDataset = {
      name: datasetName,
      files: {},
    };
    await Promise.all(
      theFiles.map(async (file, i) => {
        const formData = new FormData();
        formData.append("file", file);
        const res = await makeRequest<MynahFile>(
          "POST",
          "/api/v1/upload",
          formData,
          "multipart/form-data"
        ).catch((err) => {
          alert("something went wrong with file upload");
          console.log(err);
        });
        if (!res) return;
        dataset.files[res.uuid] = file.webkitRelativePath.split("/")[1];
        setFiles((filesState) => {
          if (!filesState) return;
          filesState.splice(i, 1, { file, isFinished: true });
          const numFinished = filesState.filter(
            ({ isFinished }) => isFinished
          ).length;
          setNumFinished(numFinished);
          return filesState;
        });
      })
    );
    await makeRequest<MynahICDataset>(
      "POST",
      "/api/v1/dataset/ic/create",
      JSON.stringify(dataset),
      "application/json"
    )
      .then((res) => {
        // alert('making dataset') // this still gets called even if closed - yay
        // can potentially use res to update list without refetching, but idk refetches aren't the end of the world..
        refetch();
      })
      .catch((err) => {
        alert("error making dataset");
        console.log(err);
      });
  };

  const onClose = () => {
    setNumFinished(0);
    setFiles(undefined);
    setSelectedType(undefined);
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
            onChange={(e) => setDatasetName(e.target.value)}
          />
          <Listbox
            value={selectedType}
            onChange={setSelectedType}
            as="div"
            className="relative"
          >
            <Listbox.Button
              className={clsx(
                "w-full px-[10px] border border-grey1 h-[56px] items-center text-left flex mt-[10px]",
                !selectedType && "text-grey2"
              )}
            >
              {selectedType ? selectedType.name : "Data type"}
              <img
                src={ArrowIcon}
                alt="arrow"
                className="h-[10px] ml-auto my-auto"
              />
            </Listbox.Button>
            <Listbox.Options className="rounded-b-[5px] overflow-hidden absolute w-full divide-y divide-grey2">
              {options.map((option) => (
                <Listbox.Option
                  className="w-full text-black pl-[10px] h-[40px] bg-grey1 hover:bg-sideBar hover:text-white flex items-center"
                  key={option.id}
                  value={option}
                >
                  {option.name}
                </Listbox.Option>
              ))}
            </Listbox.Options>
          </Listbox>

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
            directory=""
            webkitdirectory=""
            onChange={(e) => {
              const uploadedFiles = e.target.files;
              if (uploadedFiles === null) return;
              uploadFiles(Array.from(uploadedFiles));
            }}
          />
        </form>
        {files ? (
          <div className="overflow-y-scroll max-h-[500px] w-full">
            {files.map(({ file, isFinished }) => {
              const filename = file.name;
              const src = URL.createObjectURL(file);
              return (
                <div className="flex w-full h-[60px] items-center border-b border-grey1">
                  <img className="h-[40px] aspect-square mr-[10px]" src={src} />
                  {filename}
                  <div
                    className={clsx(
                      "animate-spin border-t border-b border-l border-sidebarSelected h-[24px] aspect-square rounded-full ml-auto mr-[10px] my-auto",
                      isFinished && "hidden"
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
            onClick={() => document.getElementById("upload")?.click()}
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
          {isUploadFinished
            ? "Upload complete"
            : "We will let you know once all of your data sets have been uploaded"}
        </p>
      </div>
    </Dialog>
  );
}
