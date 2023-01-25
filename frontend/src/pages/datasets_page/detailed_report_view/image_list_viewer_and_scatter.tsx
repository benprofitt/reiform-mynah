import { useCallback, useEffect, useRef, useState } from "react";
import MyPlot from "./plot";
import ImageList from "./image_list";
import * as Plotly from "plotly.js-dist-min";
import Plot from "react-plotly.js";
import SelectedImage from "./selected_image";
import { MynahICDatasetReport } from "../../../utils/types";
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

export interface ImageListViewerAndScatterProps {
  reportData: MynahICDatasetReport;
}

export default function ImageListViewerAndScatter(
  props: ImageListViewerAndScatterProps
): JSX.Element {
  const { reportData } = props;

  const dataConversion: Partial<Plotly.ScatterData>[] = Object.entries(
    reportData.points
  ).map(([imgClassName, pointList], idx) =>
    _.reduce(
      pointList,
      (acc: Partial<Plotly.ScatterData>, point, ix) => {
        return {
          ...acc,
          // these + indexes are just to seperate the classes cuz everything was coming in at 0,0
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
            stringToColor(imgClassName.repeat(10))
          ),
          size: Array.from({ length: pointList.length }, () => 12),
          line: { width: 0 },
        },
      }
    )
  );

  const plotRef = useRef<Plot>(null);

  const [last, setLast] = useState<{
    x: Plotly.Datum;
    y: Plotly.Datum;
    pointIndex: number;
    pointClass: number;
  } | null>(null);

  const data: Partial<Plotly.ScatterData>[] = [
    ...dataConversion,
    last
      ? {
          x: [last.x],
          y: [last.y],
          type: "scattergl",
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
    showlegend: false,
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

  const xwidth = (maxx - minx) / 30;
  const ywidth = (maxy - miny) / 15;

  // this one function is called both on clicking a point on the graph
  // AND on clicking an item in the list of images. it changes the value of
  // 'last' and thus updates whats being displayed in the top right as well
  // as triggers a useeffect which scrolls to the highlighted point
  const setPoint = useCallback(
    (
      x: Plotly.Datum,
      y: Plotly.Datum,
      pointIndex: number,
      pointClass: number
    ) => {
      setLast({ x, y, pointIndex, pointClass });

      if (!plotRef.current) return;

      const layout = plotRef.current.props.layout;
      const xrange = layout.xaxis?.range?.map(Number);
      const yrange = layout.yaxis?.range?.map(Number);
      if (!xrange || !yrange) return;

      const xwidth = xrange[1] - xrange[0];
      const ywidth = yrange[1] - yrange[0];


      if (
        isInRectangle(
          Number(x),
          Number(y),
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
            Math.max(Number(x) - xwidth, minx),
            Math.min(Number(x) + xwidth, maxx),
          ],
        },
        yaxis: {
          range: [
            Math.max(Number(y) - ywidth, miny),
            Math.min(Number(y) + ywidth, maxy),
          ],
        },
      };

      setLayout((layout) => {
        return { ...layout, ...newLayout };
      });
    },
    []
  );
  // last only needs to store point class and point index because x and y can be derived from that
  const selectedPointData = last
    ? reportData.points[last.pointClass][last.pointIndex]
    : null;
  return (
    <>
      <div className="w-[30%] shadow-huge h-full">
        <ImageList
          data={data}
          setPoint={setPoint}
          last={last}
          points={reportData.points}
        />
      </div>
      {/* picture and graph */}
      <div className="w-[70%] bg-grey">
        <div className="h-[50%] px-[15px] py-[25px]">
          <SelectedImage
            last={last}
            data={data}
            selectedPointData={selectedPointData}
          />
        </div>
        <div className="h-[50%]">
          <MyPlot
            data={data}
            setPoint={setPoint}
            plotLayout={plotLayout}
            plotRef={plotRef}
          />
        </div>
      </div>
    </>
  );
}
