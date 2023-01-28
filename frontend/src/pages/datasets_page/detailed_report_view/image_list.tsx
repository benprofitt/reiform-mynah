import clsx from "clsx";
import RightArrowIcon from "../../../images/RightArrowIcon.svg";
import { CSSProperties, useEffect, useRef } from "react";
import { FixedSizeList as List } from "react-window";
import FilterDropdown from "../../../components/filter_dropdown";
import { sum } from "lodash";
import { MynahFile, MynahICDatasetReport } from "../../../utils/types";
import Image from "../../../components/Image";
import { useQuery } from "react-query";
import makeRequest from "../../../utils/apiFetch";

export interface ImageListProps {
  data: Partial<Plotly.ScatterData>[];
  setPoint: (
    pointIndex: number,
    pointClass: string
  ) => void;
  selectedPoint: {
    pointIndex: number;
    pointClass: string;
  } | null;
  points: MynahICDatasetReport["points"]
}

interface RowProps {
  index: number;
  style: CSSProperties;
  onClick: () => void;
  selected: boolean;
  imgLoc: string;
  fileName: string;
}

function Row(props: RowProps): JSX.Element {
  const { index, style, onClick, selected, imgLoc, fileName } = props;
  return (
    <div
      className={clsx(
        "border-b-4 border-grey h-[80px] flex items-center px-[30px]",
        selected && "bg-linkblue bg-opacity-5"
      )}
      style={style}
      onClick={onClick}
    >
      <Image className="h-[47px] aspect-square mr-[10px]" src={imgLoc} /> {fileName}
      <img src={RightArrowIcon} className="ml-auto" />
    </div>
  );
}

export default function ImageList(props: ImageListProps): JSX.Element {
  const { data, setPoint, selectedPoint, points } = props;
  const allButLast = data.slice(0, -1);
  const xList = allButLast.map((val) => val.x).flat();
  const yList = allButLast.map((val) => val.y).flat();
  const classLens = allButLast.map((val) => val.x?.length);
  const allIds: string[] = Object.entries(
    points
  ).flatMap(([imgClassName, pointList], idx) =>
    pointList.flatMap((x) => x.fileid))

  const pointClasses = Object.keys(points)

  const query = `/api/v1/file/list?fileid=${allIds.join("&fileid=")}`;
  const { error, isLoading, data: fileData } = useQuery("datasetFiles", () =>
    makeRequest<{ [fileId: string]: MynahFile }>("GET", query)
  );
  const listRef = useRef<List | null>();

  const headerid = "header";

  const classNum = selectedPoint ? Object.keys(points).indexOf(selectedPoint?.pointClass) : 0

  const lastFlatIndex = selectedPoint
    ? sum(classLens.slice(0, classNum)) + selectedPoint.pointIndex
    : 0;

  useEffect(() => {
    if (!selectedPoint || !listRef.current) return;
    listRef.current.scrollToItem(lastFlatIndex, "smart");
  }, [selectedPoint]);

  console.log("imagelist render");
  if (!xList || !yList) return <></>;

  return (
    <>
      <div
        className="h-[110px] py-[20px] px-[30px] border-b-4 border-grey"
        id={headerid}
      >
        <h3 className="mb-[10px] text-[20px] font-medium">Images</h3>
        <FilterDropdown leftAligned />
      </div>
      <List
        className="no-scrollbar"
        height={window.innerHeight - 110}
        itemCount={xList.length}
        itemSize={80}
        ref={(el) => (listRef.current = el)}
        width={document.getElementById(headerid)?.clientWidth ?? 500}
      >
        {(props) => {
          let index = props.index;
          let classNum = 0;
          while (index - (classLens[classNum] ?? 0) >= 0) {
            index -= classLens[classNum] ?? 0;
            classNum += 1;
          }
          const className = pointClasses[classNum]
          const pointData = points[className][index]
          const imgLoc = pointData ? `/api/v1/file/${pointData.fileid}/${pointData.image_version_id}` : ''
          const fileName = pointData && fileData ? fileData[pointData.fileid].name : ''
          return (
            // maybe we want to memoize rows? 
            // https://react-window.vercel.app/#/api/areEqual
            // maybe memoizing would be good or maybe not..
            <Row
              imgLoc={imgLoc}
              // will add props to send over data to get the file name and image
              fileName={fileName}
              index={props.index}
              style={props.style}
              selected={selectedPoint !== null && props.index === lastFlatIndex}
              onClick={() => {
                setPoint(
                  index,
                  className
                );
              }}
            />
          );
        }}
      </List>
    </>
  );
}
