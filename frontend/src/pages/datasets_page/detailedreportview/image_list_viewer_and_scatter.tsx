
import { useCallback, useEffect, useRef, useState } from "react";
import MyPlot from "./plot";
import ImageList from "./image_list";
import * as Plotly from "plotly.js-dist-min";
import Plot from "react-plotly.js";
import SelectedImage from "./selected_image";

const defaultColors = Array.from({ length: 80000 }, () => "red");
const defaultSizes = Array.from({ length: 80000 }, () => 10);
const randomdata: Partial<Plotly.ScatterData>[] = [
  {
    // selectedpoints: [0],
    x: Array.from({ length: 80000 }, (v, i) => i),
    y: Array.from(
      { length: 80000 },
      (v, i) => ((i - 40000) / 1000) ** 2 + Math.random() * 1000
    ),
    type: "scattergl",
    // selected: {},
    mode: "markers",
    marker: { color: defaultColors, size: defaultSizes, line: { width: 0 } },
  },
  {
    x: [],
    y: [],
    type: "scattergl",
    mode: "markers",
    marker: { color: "black", size: 15 },
  },
];

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

export interface ImageListViewerAndScatterProps {}

export default function ImageListViewerAndScatter(
  props: ImageListViewerAndScatterProps
): JSX.Element {
  const plotRef = useRef<Plot>(null);

  // need to set these based on data
  // they are for following clicked on points
  const [xmin, xmax, ymin, ymax, xwidth, ywidth] = [
    0, 80000, 0, 2600, 3000, 200,
  ];
  // maybe xwidth and ywidth should be determined programatically based on current zoom, or maybe just zoom all the way out

  const [last, setLast] = useState<{
    x: Plotly.Datum;
    y: Plotly.Datum;
    pointIndex: number;
  } | null>(null);

  const [data, setData] = useState<Partial<Plotly.ScatterData>[]>(randomdata);

  const [plotLayout, setLayout] = useState<Partial<Plotly.Layout>>({
    showlegend: false,
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    xaxis: {
      title: 'x-axis label'
    },
    yaxis: {
      title: 'y-axis label'
    }
  });

  // it wasn't interactable until set data was called for the first time idk why
  // we will need to do this with the real data from the api
  useEffect(() => {
    if (!plotRef.current) return;
    const update: Partial<Plotly.ScatterData> = {
      x: [],
      y: [],
    };
    const newdata: Partial<Plotly.ScatterData>[] = [
      randomdata[0],
      { ...randomdata[1], ...update },
    ];
    setData(newdata);
  }, []);

  // this one function is called both on clicking a point on the graph
  // AND on clicking an item in the list of images. it changes the value of
  // 'last' and thus updates whats being displayed in the top right as well
  // as triggers a useeffect which scrolls to the highlighted point
  const setPoint = useCallback(
    (x: Plotly.Datum, y: Plotly.Datum, pointIndex: number) => {
      setLast({ x, y, pointIndex });

      const update: Partial<Plotly.ScatterData> = {
        x: [x],
        y: [y],
      };
      const newdata: Partial<Plotly.ScatterData>[] = [
        randomdata[0],
        { ...randomdata[1], ...update },
      ];
      setData(newdata);

      if (!plotRef.current) return;

      const layout = plotRef.current.props.layout;
      const xrange = layout.xaxis?.range;
      const yrange = layout.yaxis?.range;

      if (!xrange || !yrange) return;

      if (
        isInRectangle(
          Number(x),
          Number(y),
          Number(xrange[0]),
          Number(xrange[1]),
          Number(yrange[0]),
          Number(yrange[1])
        )
      )
        return;
      const newLayout: Partial<Plotly.Layout> = {
        xaxis: {
          range: [
            Math.max(Number(x) - xwidth, xmin),
            Math.min(Number(x) + xwidth, xmax),
          ],
        },
        yaxis: {
          range: [
            Math.max(Number(y) - ywidth, ymin),
            Math.min(Number(y) + ywidth, ymax),
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
        <ImageList data={data} setPoint={setPoint} last={last} />
      </div>
      {/* picture and graph */}
      <div className="w-[70%] bg-grey">
        <div className="h-[50%] px-[15px] py-[25px]">
          <SelectedImage last={last} />
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
