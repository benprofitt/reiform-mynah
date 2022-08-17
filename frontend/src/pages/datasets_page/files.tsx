import { Dialog, Menu } from "@headlessui/react";
import clsx from "clsx";
import {
  MynahFile,
  MynahICDataset,
  MynahICFile,
} from "../../utils/types";
import { useQuery } from "react-query";
import makeRequest from "../../utils/apiFetch";
import Image from "../../components/Image";
import getDate from "../../utils/date";
import { useState } from "react";

const File = (props: {
  version: string;
  fileId: string;
  file: MynahICFile;
  data: MynahFile;
  onClick: () => void;
}): JSX.Element => {
  const { version, fileId, file, data, onClick } = props;
  const { image_version_id } = file;
  const imgClass = "current_class" in file ? file.current_class : "OD class";
  if (!data) return <></>;
  const { date_created } = data;
  return (
    <div
      className="w-full hover:shadow-floating cursor-pointer bg-white border border-grey4 rounded-md text-[18px] flex items-center mt-[10px] relative text-left h-[72px]"
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
    </div>
  );
};

export interface FilesProps { dataset: MynahICDataset }

export default function Files(props: FilesProps): JSX.Element {
  const { dataset } = props;
  const [selectedFile, setSelectedFile] = useState<number | null>(null);
  const files = Object.entries(dataset.versions)
    .reverse()
    .map(([version, data]: [string, MynahICDataset['versions']['0']]) =>
      Object.entries(data.files).map(
        ([fileId, file]: [string, MynahICFile]) => ({
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
    file: MynahICFile;
  }[];
  data: Record<string, MynahFile>;
}) => {
  const { selected, setSelected, files, data } = props;
  const theFile = selected !== null ? files[selected] : null;
  const fileData = theFile !== null ? data[theFile.fileId] : null;

  if (theFile === null) return <></>;
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
              <MetaDetail title="Class" value={theFile.file.current_class} />
              <MetaDetail
                title="Original Class"
                value={theFile.file.original_class}
              />
            </div>
            {/* {JSON.stringify({ ...theFile, fileData })} */}
          </div>
        </div>
      </main>
    </Dialog>
  );
};