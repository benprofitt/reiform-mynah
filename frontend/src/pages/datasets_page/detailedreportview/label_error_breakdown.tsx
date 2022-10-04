import { Tab } from "@headlessui/react";
import clsx from "clsx";
import Plot from "react-plotly.js";

export default function LabelErrorBreakdown(): JSX.Element {
  return (
    <>
      <h3 className="font-black text-[20px]">Breakdown</h3>
      <Tab.Group>
        <Tab.List>
          {["Table", "Graph"].map((title) => (
            <Tab key={title} className="focus:outline-none">
              {({ selected }) => (
                <div
                  className={clsx(
                    "relative h-[40px] mr-[20px] mt-[20px] font-bold uppercase",
                    selected ? "text-linkblue" : "text-grey2"
                  )}
                >
                  {title}
                  {selected && (
                    <div className="absolute bottom-0 w-full bg-linkblue h-[5px] rounded-sm"></div>
                  )}
                </div>
              )}
            </Tab>
          ))}
        </Tab.List>
        <Tab.Panels className="pt-[20px]">
          <Tab.Panel>
            <table className="w-full">
              <th className="text-left">Class 1</th>
              <tr>
                <td>Bad:</td>
                <td className="text-right">129</td>
              </tr>
              <tr>
                <td>Acceptable:</td>
                <td className="text-right">905</td>
              </tr>
              <tr>
                <td>Total:</td>
                <td className="text-right">1235</td>
              </tr>
            </table>
          </Tab.Panel>
          <Tab.Panel>
            {() => {
              const trace1 = {
                x: [129, 229],
                y: ["class1", "class2"],
                name: "Mislabeled",
                text: "Mislabeled",
                orientation: "h",
                marker: {
                  color: "rgba(55,128,191,0.6)",
                  width: 1,
                },
                type: "bar",
              };

              const trace2 = {
                x: [905, 805],
                y: ["class1", "class2"],
                name: "Correct",
                text: "Correct",
                orientation: "h",
                type: "bar",
                marker: {
                  color: "rgba(255,153,51,0.6)",
                  width: 1,
                },
              };

              return (
                <Plot
                  className="w-full"
                  data={[trace1, trace2] as Partial<Plotly.Data>[]}
                  layout={{ showlegend: false, barmode: "group", bargap: 10 }}
                  config={{
                    responsive: true, // removing this makes it so the graph doesn't move when things change (the same way that relayouting kept it in the same place)
                    displayModeBar: false,
                    // modeBarButtonsToRemove: ["lasso2d", "autoScale2d", "select2d"],
                  }}
                />
              );
            }}
          </Tab.Panel>
        </Tab.Panels>
      </Tab.Group>
    </>
  );
}
