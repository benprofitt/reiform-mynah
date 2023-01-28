import { useCallback, useEffect, useRef, useState } from "react";
import MyPlot from "./plot";
import ImageList from "./image_list";
import * as Plotly from "plotly.js-dist-min";
import Plot from "react-plotly.js";
import SelectedImage from "./selected_image";
import { MynahICDatasetReport, MynahICProcessTaskCorrectMislabeledImagesReport, MynahICProcessTaskDiagnoseMislabeledImagesReport, MynahICProcessTaskReportMetadata, MynahICProcessTaskType } from "../../../utils/types";
import _ from "lodash";

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

function getMislabeledImages(reportMetadata: MynahICProcessTaskReportMetadata, reportType: MynahICProcessTaskType): string[] {

  if (reportType == 'ic::correct::mislabeled_images') {
    return Object.values((reportMetadata as MynahICProcessTaskCorrectMislabeledImagesReport).class_label_errors).flatMap((value) => [...value.mislabeled_corrected, ...value.mislabeled_removed])
  }

  if (reportType == 'ic::diagnose::mislabeled_images') {
    return Object.values((reportMetadata as MynahICProcessTaskDiagnoseMislabeledImagesReport).class_label_errors).flatMap((value) => [...value.mislabeled])
  }

  return []
}

const colors = ['red', 'blue', 'green', 'yellow']

export interface ImageListViewerAndScatterProps {
  reportData: MynahICDatasetReport;
  reportType: MynahICProcessTaskType
}

export default function ImageListViewerAndScatter(
  props: ImageListViewerAndScatterProps
): JSX.Element {
  const { reportData, reportType } = props;

  const taskMetaData = reportData.tasks.filter(({ type }) => type == reportType)[0].metadata;

  const dataConversion: Partial<Plotly.ScatterData>[] = Object.entries(
    reportData.points
  ).map(([imgClassName, pointList], idx) =>
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
          color: Array.from({ length: pointList.length }, () =>
            idx < colors.length ? colors[idx] : stringToColor(imgClassName.repeat(10))
          ),
          size: Array.from({ length: pointList.length }, () => 12),
          line: { width: 0 },
        },
      }
    )
  );

  const mislabled_images = getMislabeledImages(taskMetaData, reportType)

  const plotRef = useRef<Plot>(null);

  const [selectedPoint, setSelectedPoint] = useState<{
    pointIndex: number;
    pointClass: string;
  } | null>(null);


  const selectedPointData = selectedPoint
    ? reportData.points[selectedPoint.pointClass][selectedPoint.pointIndex]
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
            size: 12,
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
      title: "x-axis label",
    },
    yaxis: {
      title: "y-axis label",
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
  const setPoint = useCallback(
    (
      pointIndex: number,
      pointClass: string
    ) => {
      setSelectedPoint({ pointIndex, pointClass });

      const newLastData = reportData.points[pointClass][pointIndex]
      if (!plotRef.current) return;

      const {x,y} = newLastData

      const layout = plotRef.current.props.layout;
      const xrange = layout.xaxis?.range?.map(Number);
      const yrange = layout.yaxis?.range?.map(Number);
      if (!xrange || !yrange) return;

      const xwidth = xrange[1] - xrange[0];
      const ywidth = yrange[1] - yrange[0];


      if (
        isInRectangle(
          x,
          y,
          xrange[0],
          xrange[1],
          yrange[0],
          yrange[1]
        )
      )
        return;
      const newLayout: Partial<Plotly.Layout> = {
        xaxis: {
          range: [
            Math.max(x - xwidth, minx),
            Math.min(x + xwidth, maxx),
          ],
        },
        yaxis: {
          range: [
            Math.max(y - ywidth, miny),
            Math.min(y + ywidth, maxy),
          ],
        },
      };

      setLayout((layout) => {
        return { ...layout, ...newLayout };
      });
    },
    []
  );
  
  return (
    <>
      <div className="w-[30%] shadow-huge h-full">
        <ImageList
          data={data}
          setPoint={setPoint}
          selectedPoint={selectedPoint}
          points={reportData.points}
        />
      </div>
      {/* picture and graph */}
      <div className="w-[70%] bg-grey">
        <div className="h-[50%] px-[15px] py-[25px]">
          <SelectedImage
            selectedPoint={selectedPoint}
            data={data}
            selectedPointData={selectedPointData}
            mislabeledImages={mislabled_images}
          />
        </div>
        <div className="h-[50%]">
          <MyPlot
            data={data}
            classNames={Object.keys(reportData.points)}
            setPoint={setPoint}
            plotLayout={plotLayout}
            plotRef={plotRef}
          />
        </div>
      </div>
    </>
  );
}
