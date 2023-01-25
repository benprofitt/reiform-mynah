import clsx from "clsx";
import Plot from "react-plotly.js";

const randomdata: Partial<Plotly.ScatterData>[] = [
  {
    // loading it up as scattergl def slowed it down
    x: Array.from({ length: 30 }, (v, i) => i),
    y: Array.from({ length: 30 }, (v, i) => Math.random() * 3 + i ),
    type: "scatter",
  },
  {
    x: Array.from({ length: 30 }, (v, i) => i),
    y: Array.from({ length: 30 }, (v, i) => Math.random() * 2 + 30 - i ),
    type: "scatter",
    marker: { color: "black", size: 15 },
  },
];

const layout: Partial<Plotly.Layout> = {
  showlegend: false,
  title: "Loss per Epoch",
  plot_bgcolor: "rgba(0,0,0,0)",
  paper_bgcolor: "rgba(0,0,0,0)",
  xaxis: {
    title: "x-axis label",
  },
  yaxis: {
    title: "y-axis label",
  },
};

const tableData = {
  "Class 1": {
    Recall: 0.129,
    Precision: 0.905,
    "F1-Score": 0.798,
  },
  "Class 2": {
    Recall: 0.129,
    Precision: 0.905,
    "F1-Score": 0.798,
  },
  "Class 3": {
    Recall: 0.129,
    Precision: 0.905,
    "F1-Score": 0.798,
  },
};

export default function Results(): JSX.Element {
  return (
    <div className="grid grid-cols-2 h-full">
      <div className="w-[50%]">
        {Object.entries(tableData).map(([key, value], ix) => (
          <div className="border-b-2 pb-[20px] pt-[20px]">
          <table className={clsx("w-full max-w-[500px]")}>
            <th className="text-left pb-[10px]">{key}</th>
            {Object.entries(value).map(([key, value]) => (
              <>
                <tr>
                  <td>{key}</td>
                  <td className="text-right">{value}</td>
                </tr>
              </>
            ))}
          </table>
          </div>
        ))}
      </div>
      <div className="h-full">
        <Plot
          className="w-full h-full"
          data={randomdata}
          layout={layout}
          config={{
            scrollZoom: true,
            responsive: true, // removing this makes it so the graph doesn't move when things change (the same way that relayouting kept it in the same place)
            displayModeBar: false,
            // modeBarButtonsToRemove: ["lasso2d", "autoScale2d", "select2d"],
          }}
        />
      </div>
    </div>
  );
}
