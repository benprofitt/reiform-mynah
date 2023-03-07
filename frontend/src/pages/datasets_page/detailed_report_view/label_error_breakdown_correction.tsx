import { Tab } from "@headlessui/react";
import clsx from "clsx";
import Plot from "react-plotly.js";
import { MynahICProcessTaskCorrectMislabeledImagesReport } from "../../../utils/types";

export interface LabelErrorBreakdownCorrectionProps {
  taskMetadata: MynahICProcessTaskCorrectMislabeledImagesReport;
}

export default function LabelErrorBreakdownDiagnosis(
  props: LabelErrorBreakdownCorrectionProps
): JSX.Element {
  const { taskMetadata } = props;
  
  const correctedCounts = Object.values(taskMetadata.class_label_errors).map(
    ({ mislabeled_corrected }) => mislabeled_corrected.length
  );

  const removedCounts = Object.values(taskMetadata.class_label_errors).map(
    ({ mislabeled_removed }) => mislabeled_removed.length
  );

  const unchangedCounts = Object.values(taskMetadata.class_label_errors).map(
    ({ unchanged }) => unchanged.length
  );

  const classNames = Object.keys(taskMetadata.class_label_errors).map(
    (name) => "Class " + name
  );

  const trace1 = {
    x: correctedCounts,
    y: classNames,
    name: "Mislabeled Corrected",
    text: "Mislabeled Corrected",
    orientation: "h",
    marker: {
      color: "rgba(55,128,191,0.6)",
      width: 1,
    },
    type: "bar",
  };

  const trace2 = {
    x: removedCounts,
    y: classNames,
    name: "Mislabeled Removed",
    text: "Mislabeled Removed",
    orientation: "h",
    type: "bar",
    marker: {
      color: "rgba(255,153,51,0.6)",
      width: 1,
    },
  };


  const trace3 = {
    x: unchangedCounts,
    y: classNames,
    name: "Unchanged",
    text: "Unchanged",
    orientation: "h",
    type: "bar",
    marker: {
      color: "rgba(180,20,51,0.6)",
      width: 1,
    },
  };

  return (
    <>
      <h3 className="font-black text-[20px]">Breakdown</h3>
      <Tab.Group>
        <Tab.List>
          {["Table", "Chart"].map((title) => (
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
              {Object.entries(taskMetadata.class_label_errors).map(
                ([className, { mislabeled_corrected, mislabeled_removed, unchanged }]) => (
                  <>
                    <thead>
                      <tr>
                        <th className="text-left">Class {className}</th>
                      </tr>
                    </thead>
                    <tbody className="border-b-2">
                      <tr>
                        <td>Unchanged:</td>
                        <td className="text-right">{unchanged.length}</td>
                      </tr>
                      <tr>
                        <td>Mislabeled Corrected:</td>
                        <td className="text-right">{mislabeled_corrected.length}</td>
                      </tr>
                      <tr>
                        <td>Mislabeled Removed:</td>
                        <td className="text-right">
                          {mislabeled_removed.length}
                        </td>
                      </tr>
                      <tr className="font-bold">
                        <td>Total:</td>
                        <td className="text-right">
                          {mislabeled_corrected.length + unchanged.length}
                        </td>
                      </tr>
                    </tbody>
                  </>
                )
              )}
            </table>
          </Tab.Panel>
          <Tab.Panel>
            <Plot
              className="w-full"
              data={[trace1, trace2, trace3] as Partial<Plotly.Data>[]}
              layout={{ showlegend: false, barmode: "group", bargap: 10 }}
              config={{
                responsive: true, // removing this makes it so the graph doesn't move when things change (the same way that relayouting kept it in the same place)
                displayModeBar: false,
                // modeBarButtonsToRemove: ["lasso2d", "autoScale2d", "select2d"],
              }}
            />
          </Tab.Panel>
        </Tab.Panels>
      </Tab.Group>
    </>
  );
}
