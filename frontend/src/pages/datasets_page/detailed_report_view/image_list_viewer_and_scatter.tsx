import { useCallback, useMemo, useRef, useState } from "react";
import MyPlot from "./plot";
import ImageList from "./image_list";
import * as Plotly from "plotly.js-dist-min";
import Plot from "react-plotly.js";
import SelectedImage from "./selected_image";
import {
  MynahICDatasetReport,
  MynahICPoint,
  MynahICProcessTaskCorrectMislabeledImagesReport,
  MynahICProcessTaskDiagnoseMislabeledImagesReport,
  MynahICProcessTaskReportMetadata,
  MynahICProcessTaskType,
  MynahICTaskReport,
} from "../../../utils/types";
import _ from "lodash";

export type MislabeledType = 'mislabeled' | 'mislabeled_removed' | 'mislabeled_corrected' | 'unchanged'

function stringToColor(str: string): string {
  var hash = 0;
  for (var i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  var colour = "#";
  for (var i = 0; i < 3; i++) {
    var value = (hash >> (i * 8)) & 0xff;
    colour += ("00" + value.toString(16)).substr(-2);
  }
  return colour;
}

function isInRectangle(
  x: number,
  y: number,
  xmin: number,
  xmax: number,
  ymin: number,
  ymax: number
): boolean {
  console.log({ x, y, xmin, xmax, ymin, ymax });
  return x >= xmin && x <= xmax && y >= ymin && y <= ymax;
}

function getMislabledFileIdDict(reportMetadata: MynahICProcessTaskReportMetadata, reportType: MynahICProcessTaskType): { [fileId: string]: MislabeledType } {
  const dict: { [fileId: string]: MislabeledType } = {}
  if (reportType == 'ic::correct::mislabeled_images') {
    Object.values((reportMetadata as MynahICProcessTaskCorrectMislabeledImagesReport).class_label_errors).forEach(({ mislabeled_corrected, mislabeled_removed, unchanged }) => {
      mislabeled_corrected.forEach((fileId) => dict[fileId] = 'mislabeled_corrected')
      mislabeled_removed.forEach((fileId) => dict[fileId] = 'mislabeled_removed')
      unchanged.forEach((fileId) => dict[fileId] = 'unchanged')
    })
  }
  if (reportType == 'ic::diagnose::mislabeled_images') {
    Object.values((reportMetadata as MynahICProcessTaskDiagnoseMislabeledImagesReport).class_label_errors).forEach(({ mislabeled, correct }) => {
      correct.forEach((fileId) => dict[fileId] = 'unchanged')
      mislabeled.forEach((fileId) => dict[fileId] = 'mislabeled')
    })
  }
  return dict
}

function getMislabeledType(reportMetadata: MynahICProcessTaskReportMetadata, reportType: MynahICProcessTaskType, fileId: string, className: string): MislabeledType {
  if (!reportType.endsWith('mislabeled_images')) {
    return 'unchanged'
  }
  if (reportType == "ic::correct::mislabeled_images") {
    const { mislabeled_corrected, mislabeled_removed } = (reportMetadata as MynahICProcessTaskCorrectMislabeledImagesReport).class_label_errors[className]
    if (mislabeled_corrected.includes(fileId)) {
      return 'mislabeled_corrected'
    }
    if (mislabeled_removed.includes(fileId)) {
      return 'mislabeled_removed'
    }
    return 'unchanged'
  }
  const { mislabeled } = (reportMetadata as MynahICProcessTaskDiagnoseMislabeledImagesReport).class_label_errors[className]
  return mislabeled.includes(fileId) ? 'mislabeled' : 'unchanged'
}
// function getMislabeledImages(
//   reportMetadata: MynahICProcessTaskReportMetadata,
//   reportType: MynahICProcessTaskType
// ): string[] {
//   if (!reportType.endsWith('mislabeled_images')) {
//     return []
//   }
//   if (reportType == "ic::correct::mislabeled_images") {
//     return Object.values(
//       (reportMetadata as MynahICProcessTaskCorrectMislabeledImagesReport)
//         .class_label_errors
//     ).flatMap((value) => [
//       ...value.mislabeled_corrected,
//       ...value.mislabeled_removed,
//     ]);
//   }

//   return Object.values(
//       (reportMetadata as MynahICProcessTaskDiagnoseMislabeledImagesReport)
//         .class_label_errors
//     ).flatMap((value) => [...value.mislabeled]);
// }

const colors = ["red", "blue", "green", "yellow"];

const allowedMislabeledTypesDiagnosis: MislabeledType[] = ['mislabeled', 'unchanged']

const allowedMislabeledTypesCorrection: MislabeledType[] = ['mislabeled_corrected', 'mislabeled_removed', 'unchanged']

export interface ImageListViewerAndScatterProps {
  reportData: MynahICDatasetReport;
  reportType: MynahICProcessTaskType;
}

const plotlyDotSize = 11

export default function ImageListViewerAndScatter(
  props: ImageListViewerAndScatterProps
): JSX.Element {
  const { reportData, reportType } = props;

  const taskMetaData = reportData.tasks.filter(
    ({ type }) => type == reportType
  )[0].metadata;

  const fileIdMislabeledTypeMap = useMemo(() => getMislabledFileIdDict(taskMetaData, reportType), [taskMetaData])
  
  const allowedMislabeledTypes = reportType == 'ic::correct::mislabeled_images' ? allowedMislabeledTypesCorrection : allowedMislabeledTypesDiagnosis
  const allIds = Object.values(reportData.points).flatMap((x) => x.map((y) => y.fileid))
  // const mislabled_images = getMislabeledImages(taskMetaData, reportType);
  const [filteredClasses, setFilteredClasses] = useState<string[]>(
    Object.keys(reportData.points)
  );
  const [mislabeledFilter, setMislabeledFilter] = useState<MislabeledType[]>(allowedMislabeledTypes);
  // what i should do instead of calling getMislabeledType on each point is take reportdata and reduce it to a map of fileId to mislabeled type
  const points: [string, MynahICPoint[]][] = Object.entries(reportData.points)
    .filter(([imgClassName]) =>
        filteredClasses.includes(imgClassName)
    )
    .map(([imgClassName, pointList]) => [
      imgClassName,
      pointList.filter(({ fileid }) =>
        mislabeledFilter
        .includes(fileIdMislabeledTypeMap[fileid])
        // .includes(getMislabeledType(taskMetaData, reportType, fileid, imgClassName))
      ),
    ]);
  const classNames = points.map(([className, pointList]) => className);
  const dataConversion: Partial<Plotly.ScatterData>[] = points.map(
    ([imgClassName, pointList], idx) =>
      _.reduce(
        pointList,
        (acc: Partial<Plotly.ScatterData>, point, ix) => {
          return {
            ...acc,
            x: [...(acc.x as Plotly.Datum[]), point.x],
            y: [...(acc.y as Plotly.Datum[]), point.y],
          };
        },
        {
          x: [],
          y: [],
          type: "scattergl",
          name: imgClassName,
          // selected: {},
          mode: "markers",
          marker: {
            // color: Array.from({ length: pointList.length }, () =>
            //   idx < colors.length
            //     ? colors[idx]
            //     : stringToColor(imgClassName.repeat(10))
            // ),
            // size: Array.from({ length: pointList.length }, () => 12),
            size: plotlyDotSize,
            line: { width: 0 },
          },
        }
      )
  );

  const plotRef = useRef<Plot>(null);

  const [selectedPoint, setSelectedPoint] = useState<{
    pointIndex: number;
    pointClass: string;
  } | null>(null);

  const selectedPointData = selectedPoint
    ? points[classNames.indexOf(selectedPoint.pointClass)][1][
        selectedPoint.pointIndex
      ]
    : null;

  const data: Partial<Plotly.ScatterData>[] = [
    ...dataConversion,
    selectedPoint && selectedPointData
      ? {
          x: [selectedPointData.x],
          y: [selectedPointData.y],
          showlegend: false,
          type: "scattergl",
          name: "Selected point",
          // selected: {},
          mode: "markers",
          marker: {
            color: "#000000",
            size: plotlyDotSize,
            line: { width: 0 },
          },
        }
      : {},
  ];

  const [plotLayout, setLayout] = useState<Partial<Plotly.Layout>>({
    showlegend: true,
    plot_bgcolor: "rgba(0,0,0,0)",
    paper_bgcolor: "rgba(0,0,0,0)",
    xaxis: {
      title: "x-axis",
    },
    yaxis: {
      title: "y-axis",
    },
  });

  const allpoints = Object.values(reportData.points).flat();

  const xes = allpoints.map((point) => point.x);
  const yes = allpoints.map((point) => point.y);
  const minx = Math.min(...xes);
  const maxx = Math.max(...xes);
  const miny = Math.min(...yes);
  const maxy = Math.max(...yes);

  // this one function is called both on clicking a point on the graph
  // AND on clicking an item in the list of images. it changes the value of
  // 'last' and thus updates whats being displayed in the top right as well
  // as triggers a useeffect which scrolls to the highlighted point
  const setPoint = useCallback((pointIndex: number, pointClass: string) => {
    setSelectedPoint({ pointIndex, pointClass });

    const newLastData = reportData.points[pointClass][pointIndex];
    if (!plotRef.current) return;

    const { x, y } = newLastData;

    const layout = plotRef.current.props.layout;
    const xrange = layout.xaxis?.range?.map(Number);
    const yrange = layout.yaxis?.range?.map(Number);
    if (!xrange || !yrange) return;

    const xwidth = xrange[1] - xrange[0];
    const ywidth = yrange[1] - yrange[0];

    if (isInRectangle(x, y, xrange[0], xrange[1], yrange[0], yrange[1])) return;
    const newLayout: Partial<Plotly.Layout> = {
      xaxis: {
        range: [Math.max(x - xwidth, minx), Math.min(x + xwidth, maxx)],
      },
      yaxis: {
        range: [Math.max(y - ywidth, miny), Math.min(y + ywidth, maxy)],
      },
    };

    setLayout((layout) => {
      return { ...layout, ...newLayout };
    });
  }, []);

  return (
    <>
      <div className="w-[30%] shadow-huge h-full">
        <ImageList
          data={data}
          setPoint={setPoint}
          selectedPoint={selectedPoint}
          points={points}
          allClassNames={Object.keys(reportData.points)}
          updateClassNamesFilter={(className: string) => {
            setSelectedPoint(null);
            setFilteredClasses((filteredClasses) =>
              filteredClasses.includes(className)
                ? filteredClasses.filter((x) => className !== x)
                : [...filteredClasses, className]
            );
          }}
          updateMislabeledFilter={(mislabeledType: MislabeledType) => {
            setSelectedPoint(null);
            setMislabeledFilter((mislabeledFilter) => mislabeledFilter.includes(mislabeledType)
            ? mislabeledFilter.filter((x) => mislabeledType !== x)
            : [...mislabeledFilter, mislabeledType]);
          }}
          mislabeledFilterSetting={mislabeledFilter}
          allowedMislabeledTypes={allowedMislabeledTypes}
          allIds={allIds}
        />
      </div>
      {/* picture and graph */}
      <div className="w-[70%] bg-grey">
        <div className="h-[50%] px-[15px] py-[25px]">
          <SelectedImage
            selectedPoint={selectedPoint}
            data={data}
            selectedPointData={selectedPointData}
            getMislabeledType={(fileId, className) => getMislabeledType(taskMetaData, reportType, fileId, className)}
          />
        </div>
        <div className="h-[50%]">
          <MyPlot
            data={data}
            classNames={classNames}
            setPoint={setPoint}
            plotLayout={plotLayout}
            plotRef={plotRef}
          />
        </div>
      </div>
    </>
  );
}
