import { Dialog, Menu } from "@headlessui/react";
import clsx from "clsx";
import { MynahFile, MynahICDataset, MynahICFile } from "../../../utils/types";
import { useQuery } from "react-query";
import makeRequest from "../../../utils/apiFetch";
import Image from "../../../components/Image";
import getDate from "../../../utils/date";
import _ from "lodash";
import { FixedSizeList as List } from "react-window";
import AutoSizer from "react-virtualized-auto-sizer";
import { Link, useLocation } from "wouter";
import { CSSProperties, memo, useRef } from "react";
import MenuItem from "../../../components/menu_item";
import EllipsisMenu from "../../../components/ellipsis_menu";

const FileListItem = memo(
  (props: {
    index: number;
    style: CSSProperties;
    version: string;
    fileId: string;
    file: MynahICFile;
    basePath: string;
  }): JSX.Element => {
    const { version, fileId, file, basePath, index, style } = props;

    const { image_version_id } = file;
    const imgClass = "current_class" in file ? file.current_class : "OD class";

    const query = `/api/v1/file/list?fileid=${fileId}`;
    const { error, isLoading, data } = useQuery(`datasetFile-${fileId}`, () =>
      makeRequest<{ [fileId: string]: MynahFile }>("GET", query)
    );
    console.log(data);

    if (!data || isLoading || error)
      return <div className="w-full bg-white">getting file...</div>;
    const fileData = data[fileId];
    const { date_created, name: fileName } = fileData;

    return (
      <div style={style}>
        <Link
          className="w-full hover:shadow-floating cursor-pointer bg-white border border-grey4 rounded-md text-[18px] flex items-center mt-[10px] relative text-left h-[72px]"
          to={`${basePath}/${fileId}`}
        >
          <h6 className="w-[40%] font-black text-black flex items-center">
            <Image
              src={`/api/v1/file/${fileId}/${image_version_id}`}
              className="w-[9%] min-w-[70px] aspect-square mr-[1.5%] object-cover"
            />
            {fileName}
          </h6>
          <h6 className="w-[20%]">{getDate(date_created)}</h6>
          <h6 className="w-[20%]">{imgClass}</h6>
          <h6 className="w-[20%]">{version}</h6>
          <EllipsisMenu />
        </Link>
      </div>
    );
  }
);

export interface FilesProps {
  dataset: MynahICDataset;
  basePath: string;
  fileId: string | undefined;
}

export default function Files(props: FilesProps): JSX.Element {
  const { dataset, basePath, fileId } = props;
  const versionKeys = Object.keys(dataset.versions);
  const latestVersion = Math.max(
    ...versionKeys.map((x) => Number(x))
  ).toString();
  const files = dataset.versions[latestVersion].files;
  const fileIds = Object.keys(files);
  const listRef = useRef<List | null>();

  // const allIds: string[] = _.reduce<string, string[]>(
  //   versionKeys,
  //   (prev, curr) => [...prev, ..._.keys(dataset.versions[curr].files)],
  //   []
  // );
  // console.log("all files");
  // // 'photo' now means 'unique file' in this count
  // const numFiles = allIds.length;

  const numFiles = fileIds.length;
  const fileCount = numFiles === 1 ? "1 photo" : `${numFiles} photos`;

  // const query = `/api/v1/file/list?fileid=${allIds.join("&fileid=")}`;
  // const { error, isLoading, data } = useQuery("datasetFiles", () =>
  //   makeRequest<{ [fileId: string]: MynahFile }>("GET", query)
  // );

  // if (error) return <div>error getting datasetFiles</div>;
  // if (isLoading || !data)
  //   return (
  //     <div className="animate-spin aspect-square border-l border-r border-b border-sidebarSelected border-6 rounded-full w-[20px]" />
  //   );
  return (
    <div className="text-grey2 h-full">
      <div className="flex">
        <h3>{fileCount}</h3>
      </div>
      <div className="mt-[30px] flex text-left">
        <h5 className="w-[40%]">Name</h5>
        <h5 className="w-[20%]">Date</h5>
        <h5 className="w-[20%]">Classes</h5>
        <h5 className="w-[20%]">Version</h5>
      </div>
      <AutoSizer>
        {({ height, width }) => (
          <List
            className="no-scrollbar"
            height={height}
            itemCount={numFiles}
            itemSize={82}
            ref={(el) => (listRef.current = el)}
            width={width}
          >
            {({ index, style }) => (
              <FileListItem
                key={index}
                index={index}
                style={style}
                version={latestVersion}
                file={files[fileIds[index]]}
                fileId={fileIds[index]}
                basePath={basePath}
              />
            )}
          </List>
        )}
      </AutoSizer>
      <DetailedFileView
        versions={dataset.versions}
        fileId={fileId}
        basePath={basePath}
      />
    </div>
  );
}

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
  versions: MynahICDataset["versions"];
  fileId: string | undefined;
  basePath: string;
}) => {
  const [_location, setLocation] = useLocation();
  const { versions, fileId, basePath } = props;
  const query = `/api/v1/file/list?fileid=${fileId ?? ""}`;
  const { error, isLoading, data } = useQuery(`datasetFile-${fileId}`, () =>
    makeRequest<{ [fileId: string]: MynahFile }>("GET", query)
  );
  if (fileId === undefined) return <></>;
  if (data === undefined || isLoading || error) return <></>;
  const fileData = data[fileId];
  const close = () => setLocation(basePath);

  return (
    <Dialog
      onClose={close}
      open={fileData !== undefined}
      className="fixed inset-0 w-full h-full"
    >
      <Dialog.Overlay className="w-full h-full bg-black absolute top-0 left-0 opacity-20 z-0" />
      <main className="bg-white z-10 h-full w-fit right-0 top-0 absolute px-[24px] max-w-[calc(100%-30px)]">
        <button
          className="absolute top-[15px] right-[20px] text-[40px] leading-none"
          onClick={close}
        >
          x
        </button>
        <h1 className="py-[24px] font-black text-[28px]">{fileData.name}</h1>
        {_.keys(versions).map((version) => {
          const file = versions[version].files[fileId];
          const { image_version_id, current_class, original_class } = file; // img_version_id should replace latest in the image src
          if (file === undefined) return <></>;
          return (
            <div className="flex border-grey1 border-2 h-[350px]" key={version}>
              <Image
                src={`/api/v1/file/${fileId}/${image_version_id}`}
                className="max-w-[min(60%,700px)] object-contain"
              />
              <div className="w-[376px]">
                <h3 className="text-[20px] p-[24px]">Version: {version}</h3>
                <div className="grid gap-[10px]">
                  <MetaDetail title="Class" value={current_class} />
                  <MetaDetail title="Original Class" value={original_class} />
                </div>
                {/* {JSON.stringify({ ...theFile, fileData })} */}
              </div>
            </div>
          );
        })}
      </main>
    </Dialog>
  );
};
