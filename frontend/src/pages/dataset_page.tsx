import Ellipsis from "../images/Ellipsis.svg";
import BackArrowIcon from "../images/BackArrowIcon.svg";
import { Link, RouteComponentProps } from "wouter";
import { Dialog, Menu, Tab } from "@headlessui/react";
import clsx from "clsx";
import {
  EitherData,
  EitherDataset,
  EitherFile,
  MynahFile,
} from "../utils/types";
import { useQuery } from "react-query";
import makeRequest from "../utils/apiFetch";
import Image from "../components/Image";
import { reduce } from "lodash";
import getDate from "../utils/date";
import { useState } from "react";

const MyTab = (props: { text: string }): JSX.Element => {
  const { text } = props;
  return (
    <Tab>
      {({ selected }) => (
        <div
          className={clsx(
            "relative h-[40px] mr-[20px] mt-[20px] font-bold uppercase",
            selected ? "text-black" : "text-grey2"
          )}
        >
          {text}
          {selected && (
            <div className="absolute bottom-0 w-full bg-linkblue h-[5px] rounded-sm"></div>
          )}
        </div>
      )}
    </Tab>
  );
};

const File = (props: {
  version: string;
  fileId: string;
  file: EitherFile;
  data: MynahFile;
  onClick: () => void;
}): JSX.Element => {
  const { version, fileId, file, data, onClick } = props;
  const { image_version_id } = file;
  const imgClass = "current_class" in file ? file.current_class : "OD class";
  if (!data) return <></>;
  const { date_created } = data;
  return (
    <button
      className="w-full hover:shadow-floating cursor-pointer bg-white h-fit border border-grey4 rounded-md text-[18px] flex items-center mt-[10px] relative text-left"
      onClick={onClick}
    >
      <h6 className="w-[40%] font-black text-black flex items-center">
        <Image
          src={`/api/v1/file/${fileId}/${image_version_id}`}
          className="w-[9%] min-w-[70px] aspect-square mr-[1.5%] object-cover"
        />
        {data.name}
      </h6>
      <h6 className="w-[20%]">{getDate(date_created)}</h6>
      <h6 className="w-[20%]">{imgClass}</h6>
      <h6 className="w-[20%]">{version}</h6>
      <Menu
        as="div"
        className="absolute inline-block text-left right-[15px] top-[30%]"
      >
        {({ open }) => (
          <>
            <div>
              <Menu.Button
                className={clsx(
                  "hover:bg-grey3 transition-colors duration-300 rounded-full w-[30px] aspect-square flex items-center justify-center group",
                  open ? "bg-grey3" : "bg-clearGrey3"
                )}
              >
                {[0, 1, 2].map((ix) => (
                  <div
                    key={ix}
                    className={clsx(
                      "rounded-full w-[4px] aspect-square mx-[2px] transition-colors duration-300 group-hover:bg-grey5 ",
                      open ? "bg-grey5" : "bg-grey6"
                    )}
                  />
                ))}
              </Menu.Button>
            </div>
            <Menu.Items className="z-10 absolute right-[15px] mt-[15px] w-56 origin-top-right rounded-md bg-white shadow-floating focus:outline-none">
              {/* <DotMenuItem src={OpenDatasetIcon} text="Open Dataset" />
          <DotMenuItem src={TrashIcon} text="Delete" /> */}
            </Menu.Items>
          </>
        )}
      </Menu>
    </button>
  );
};

const Files = (props: { dataset: EitherDataset }): JSX.Element => {
  const { dataset } = props;
  const [selectedFile, setSelectedFile] = useState<number | null>(null);
  const files = Object.entries(dataset.versions)
    .reverse()
    .map(([version, data]: [string, EitherData]) =>
      Object.entries(data.files).map(
        ([fileId, file]: [string, EitherFile]) => ({
          version,
          fileId,
          file,
        })
      )
    )
    .flat();
  const numFiles = files.length;
  const fileCount = numFiles === 1 ? "1 photo" : `${numFiles} photos`;

  const ids = files.map(({ fileId }) => fileId);
  const query = `/api/v1/file/list?fileid=${ids.join("&fileid=")}`;
  const { error, isLoading, data } = useQuery("datasetFiles", () =>
    makeRequest<Record<string, MynahFile>>("GET", query)
  );

  if (error) return <div>error getting datasetFiles</div>;
  if (isLoading || !data)
    return (
      <div className="animate-spin aspect-square border-l border-r border-b border-sidebarSelected border-6 rounded-full w-[20px]" />
    );
  return (
    <div className="text-grey2">
      <div className="flex">
        <h3>{fileCount}</h3>
      </div>
      <div className="mt-[30px] flex text-left">
        <h5 className="w-[40%]">Name</h5>
        <h5 className="w-[20%]">Date</h5>
        <h5 className="w-[20%]">Classes</h5>
        <h5 className="w-[20%]">Version</h5>
      </div>
      {files.map((file, ix) => (
        <File
          key={ix}
          {...file}
          data={data[file.fileId]}
          onClick={() => setSelectedFile(ix)}
        />
      ))}
      <DetailedFileView
        files={files}
        selected={selectedFile}
        setSelected={setSelectedFile}
        data={data}
      />
    </div>
  );
};

const ProcessDataModal = (props: {
  uuid: string;
  isOpen: boolean;
  close: () => void;
}): JSX.Element => {
  const { uuid, isOpen, close } = props;
  const [selected, setSelected] = useState<"correction" | "diagnosis" | null>(
    null
  );
  const [labelingErrors, setLabelingErrors] = useState(false);
  const [intraclassVariance, setIntraclassVariance] = useState(false);
  return (
    <Dialog
      open={isOpen}
      onClose={close}
      className="fixed inset-0 w-full h-full flex items-center justify-center"
    >
      <Dialog.Overlay className="w-full h-full bg-black absolute  top-0 left-0 opacity-20 z-0" />
      <main className="bg-white w-[528px] relative z-10 p-[30px]">
        <button
          className="absolute top-[15px] right-[20px] text-[40px] leading-none"
          onClick={close}
        >
          x
        </button>
        <Dialog.Title className="text-3xl">Process data</Dialog.Title>
        <form
          className="flex flex-col text-[18px]"
          onSubmit={(e) => {
            e.preventDefault();
            close();
          }}
        >
          <h2 className="font-bold my-[10px]">Select process options</h2>
          <label>
            <input
              className="mr-[10px]"
              type="radio"
              value="diagnosis"
              checked={selected === "diagnosis"}
              onChange={() => setSelected("diagnosis")}
            />
            Diagnosis
          </label>
          <label>
            <input
              className="mr-[10px]"
              type="radio"
              value="correction"
              checked={selected === "correction"}
              onChange={(e) => setSelected("correction")}
            />
            Correction
          </label>
          <label className="ml-[20px]">
            <input
              className="mr-[10px]"
              disabled={selected !== "correction"}
              type="checkbox"
              checked={selected === "correction" && labelingErrors}
              onChange={() => setLabelingErrors((cur) => !cur)}
            />
            Labeling errors
          </label>
          <label className="ml-[20px]">
            <input
              className="mr-[10px]"
              disabled={selected !== "correction"}
              type="checkbox"
              checked={selected === "correction" && intraclassVariance}
              onChange={() => setIntraclassVariance((cur) => !cur)}
            />
            Intra-class variance
          </label>
          <button
            type="submit"
            className={clsx(
              "w-full h-[40px] text-white mt-[30px] font-bold",
              selected === null ? "bg-grey1" : "bg-blue-600"
            )}
          >
            Start process
          </button>
        </form>
      </main>
    </Dialog>
  );
};

const MetaDetail = (props: { title: string; value: string }): JSX.Element => {
  const { title, value } = props;
  return (
    <div className="grid grid-cols-2 ml-[24px]">
      <h4 className="font-bold">{title}:</h4>
      <p>{value}</p>
    </div>
  );
};

const DetailedFileView = (props: {
  selected: number | null;
  setSelected: (x: number | null) => void;
  files: {
    version: string;
    fileId: string;
    file: EitherFile;
  }[];
  data: Record<string, MynahFile>;
}) => {
  const { selected, setSelected, files, data } = props;
  const theFile = selected !== null ? files[selected] : null;
  const fileData = theFile !== null ? data[theFile.fileId] : null;

  if (theFile === null) return <></>;
  const imgClass =
    "current_class" in theFile.file ? theFile.file.current_class : "OD class";
  const origClass =
    "original_class" in theFile.file ? theFile.file.current_class : "OD class";
  return (
    <Dialog
      onClose={() => setSelected(null)}
      open={selected !== null}
      className="fixed inset-0 w-full h-full"
    >
      <Dialog.Overlay className="w-full h-full bg-black absolute top-0 left-0 opacity-20 z-0" />
      <main className="bg-white z-10 h-full w-fit right-0 top-0 absolute px-[24px] max-w-[calc(100%-30px)]">
        <button
          className="absolute top-[15px] right-[20px] text-[40px] leading-none"
          onClick={() => setSelected(null)}
        >
          x
        </button>
        <h1 className="py-[24px] font-black text-[28px]">{fileData?.name}</h1>
        <div className="flex border-grey1 border-2 h-[350px]">
          <Image
            src={`/api/v1/file/${theFile.fileId}/${theFile.file.image_version_id}`}
            className="max-w-[min(60%,700px)] object-contain"
          />
          <div className="w-[376px]">
            <h3 className="text-[20px] p-[24px]">Version: {theFile.version}</h3>
            <div className="grid gap-[10px]">
              <MetaDetail title="Class" value={imgClass} />
              <MetaDetail title="Original Class" value={origClass} />
            </div>
            {/* {JSON.stringify({ ...theFile, fileData })} */}
          </div>
        </div>
      </main>
    </Dialog>
  );
};

export default function DatasetPage(
  props: RouteComponentProps<{ uuid: string }>
): JSX.Element {
  const { data: datasets } = useQuery<EitherDataset[]>("datasets", () =>
    makeRequest<EitherDataset[]>("GET", "/api/v1/dataset/list")
  );
  const { uuid } = props.params;
  const [processDataOpen, setProcessDataOpen] = useState(false);
  console.log("got the uuid", uuid);
  if (datasets === undefined)
    return <div>Unable to retrive datasets, are you logged in?</div>;
  const dataset = datasets.find((dataset) => dataset.uuid === uuid);
  if (dataset === undefined)
    return (
      <div>
        This dataset either does not exist or you do not have permission to see
        it
      </div>
    );
  const { dataset_name: name } = dataset;
  return (
    <div className="flex h-screen flex-1">
      <Tab.Group as="div" className="w-full flex flex-col">
        <header className="w-full h-fit border-b border-grey1 pl-[32px] relative bg-white pt-[46px]">
          <h1 className="font-bold text-[28px]">{name}</h1>
          <Link to="/">
            <button className="flex items-center text-linkblue absolute left-[30px] top-[20px] font-bold">
              <img src={BackArrowIcon} alt="back arrow" className="mr-[2px]" />
              Back
            </button>
          </Link>
          <div className="absolute right-[36px] top-[60px] space-x-[20px] text-[18px] h-[38px] flex items-center">
            <button className="text-linkblue h-full w-[134px] text-center font-medium">
              Add More Data
            </button>
            <button
              className="bg-linkblue text-white h-full w-[134px] text-center font-medium rounded-md"
              onClick={() => setProcessDataOpen(true)}
            >
              Process Data
            </button>
            <button className="h-full">
              <img src={Ellipsis} alt="more" />
            </button>
          </div>
          <Tab.List>
            <MyTab text="Files" />
            <MyTab text="Diagnosis" />
            <MyTab text="Cleaning" />
          </Tab.List>
        </header>
        <main className="bg-grey w-full p-[32px] flex-1 overflow-y-scroll">
          <Tab.Panels>
            <Tab.Panel>
              <Files dataset={dataset} />
              <ProcessDataModal
                isOpen={processDataOpen}
                close={() => setProcessDataOpen(false)}
                uuid={uuid}
              />
            </Tab.Panel>
            <Tab.Panel>Diagnosis Tab</Tab.Panel>
            <Tab.Panel>Cleaning Tab</Tab.Panel>
          </Tab.Panels>
        </main>
      </Tab.Group>
    </div>
  );
}
