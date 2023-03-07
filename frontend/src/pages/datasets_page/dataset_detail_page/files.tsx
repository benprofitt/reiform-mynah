import { MynahFile, MynahICFile, MynahICDataset } from "../../../utils/types";
import FileListItem from "./file_list_item";
import { FixedSizeList as List } from "react-window";
import AutoSizer from "react-virtualized-auto-sizer";
import { useEffect, useRef, useState } from "react";
import DetailedFileView from "./detailed_file_view";
import { useQuery } from "react-query";
import makeRequest from "../../../utils/apiFetch";
import ArrowIcon from "../../../images/ArrowIcon.svg";

export interface FilesProps {
  dataset: MynahICDataset;
  basePath: string;
  fileId: string | undefined;
}

type SortMethod = "class" | "name" | "class_desc" | "name_desc";

export default function Files(props: FilesProps): JSX.Element {
  const { dataset, basePath, fileId } = props;
  const [sortMethod, setSortMethod] = useState<SortMethod>("name_desc");
  const listRef = useRef<List | null>();

  const versionKeys = Object.keys(dataset.versions);
  const latestVersion = Math.max(
    ...versionKeys.map((x) => Number(x))
  ).toString();
  const files = dataset.versions[latestVersion].files;
  const fileIds = Object.keys(files);
  const unsortedFileIds = Object.keys(files);
  const query = `/api/v1/file/list?fileid=${unsortedFileIds.join("&fileid=")}`;
  const { error, isLoading, data } = useQuery(`datasetFiles`, () =>
    makeRequest<{ [fileId: string]: MynahFile }>("GET", query)
  );

  if (!data || isLoading || error || !fileIds.every((id) => data[id]))
    return <div className="w-full bg-white">getting files</div>;
  // write sort by class
  // write sort by name
  // write filter by class
  // write filter by name

  const numFiles = fileIds.length;
  const fileCount = numFiles === 1 ? "1 photo" : `${numFiles} photos`;

  fileIds.sort(sortingFunc(sortMethod, files, data));

  return (
    <div className="text-grey2 h-full">
      <div className="flex">
        <h3>{fileCount}</h3>
      </div>
      <div className="mt-[30px] flex text-left">
        <h5 className="w-[40%] flex" onClick={() => {
          if (sortMethod == 'name') {
            setSortMethod('name_desc')
          } else {
            setSortMethod('name')
          }
        }}>
          Name{" "}
          {sortMethod == "name" && (
            <img src={ArrowIcon} className="rotate-180" />
          )}{" "}
          {sortMethod == "name_desc" && <img src={ArrowIcon} />}
        </h5>
        <h5 className="w-[20%]">Date</h5>
        <h5 className="w-[20%] flex" onClick={() => {
          if (sortMethod == 'class') {
            setSortMethod('class_desc')
          } else {
            setSortMethod('class')
          }
        }}>
          Classes{" "}
          {sortMethod == "class" && (
            <img src={ArrowIcon} className="rotate-180" />
          )}
          {sortMethod == "class_desc" && <img src={ArrowIcon} />}
        </h5>
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
                fileData={data[fileIds[index]]}
              />
            )}
          </List>
        )}
      </AutoSizer>
      <DetailedFileView
        versions={dataset.versions}
        fileId={fileId}
        basePath={basePath}
        fileData={data[fileId ?? ""]}
      />
    </div>
  );
}
function sortingFunc(
  sortMethod: SortMethod,
  files: {
    [fileId: string]: MynahICFile;
  },
  data: {
    [fileId: string]: MynahFile;
  }
): (a: string, b: string) => number {
  if (sortMethod == "class") {
    return (a, b) => (files[a].current_class > files[b].current_class ? 1 : -1);
  }
  if (sortMethod == "class_desc") {
    return (a, b) => (files[a].current_class > files[b].current_class ? -1 : 1);
  }
  if (sortMethod == "name") {
    return (a, b) => (data[a].name > data[b].name ? 1 : -1);
  }
  if (sortMethod == "name_desc") {
    return (a, b) => (data[a].name > data[b].name ? -1 : 1);
  }
  return (a, b) => 1;
}
