import * as Plotly from "plotly.js-dist-min";
import { RefObject } from "react";
import Plot from "react-plotly.js";

export interface MyPlotProps {
  data: Partial<Plotly.ScatterData>[];
  setPoint: (pointIndex: number, pointClass: number) => void;
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
  const { data, setPoint, plotLayout, plotRef } = props;
  const divId = "plotDiv";

  console.log("plot render");

  return (
    <Plot
      className="w-full h-full"
      divId={divId}
      ref={plotRef}
      config={{
        scrollZoom: true,
        responsive: true, // removing this makes it so the graph doesn't move when things change (the same way that relayouting kept it in the same place)
        displayModeBar: false,
        // modeBarButtonsToRemove: ["lasso2d", "autoScale2d", "select2d"],
      }}
      data={data}
      layout={plotLayout}
      onClick={(e) => {
        const { x, y, pointIndex, curveNumber } = e.points[0];
        console.log(e.points[0]);
        console.log("click", { x, y, pointIndex, e });
        setPoint(pointIndex, curveNumber);
      }}
    />
  );
}

export default MyPlot;
// i'm not sure if the below helps
// export default memo(MyPlot);
