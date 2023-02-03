import * as Plotly from "plotly.js-dist-min";
import { RefObject } from "react";
import Plot from "react-plotly.js";

export interface MyPlotProps {
  data: Partial<Plotly.ScatterData>[];
  classNames: string[];
  setPoint: (pointIndex: number, pointClass: string) => void;
  plotLayout: Partial<Plotly.Layout>;
  plotRef: RefObject<Plot>;
}

const blankData: Plotly.Data[] = [
  {
    x: [],
    y: [],
    type: "scatter",
    mode: "markers",
    marker: { color: "blue" },
  },
];

function MyPlot(props: MyPlotProps) {
  const { data, setPoint, plotLayout, plotRef, classNames } = props;
  const divId = "plotDiv";

  return (
    <Plot
      className="w-full h-full"
      divId={divId}
      ref={plotRef}
      config={{
        displaylogo: false,
        scrollZoom: true,
        responsive: true, // removing this makes it so the graph doesn't move when things change (the same way that relayouting kept it in the same place)
        // displayModeBar: true,
        modeBarButtonsToRemove: ["lasso2d", "autoScale2d", "select2d", "toImage", "zoom2d", "pan2d", "resetScale2d", ],
      }}
      data={data}
      layout={plotLayout}
      onClick={(e) => {
        const { x, y, pointIndex, curveNumber } = e.points[0];
        setPoint(pointIndex, classNames[curveNumber]);
      }}
    />
  );
}

export default MyPlot;
// i'm not sure if the below helps
// export default memo(MyPlot);
