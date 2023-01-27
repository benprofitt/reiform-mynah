import React from "react";
import { MynahICProcessTaskCorrectClassSplittingReport } from "../../../utils/types";

export interface ClassSplittingBreakdownCorrectionProps {
  taskMetadata: MynahICProcessTaskCorrectClassSplittingReport;
}

export default function ClassSplittingBreakdownCorrection({
  taskMetadata,
}: ClassSplittingBreakdownCorrectionProps) {
  return (
    <table className="w-full text-[18px]">
      <thead>
        <tr>
          <th className="text-left text-[20px]">Class breakdown</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(taskMetadata.classes_splitting).map(
          ([className, { new_classes }]) => (
            <>
              <tr className="text-left">
                <td className="pt-[20px]">Class {className}</td>
              </tr>
              <tr>
                <td className="pt-[20px]">Number of splits:</td>
                <td className="text-right">{new_classes.length}</td>
              </tr>
              {new_classes.length > 1 && (
                <tr>
                  <td className="pt-[20px]">
                    New Class Names: {new_classes.join(", ")}
                  </td>
                </tr>
              )}
              <tr className="border-b-2" />
            </>
          )
        )}
        <tr className="border-b-2" />
      </tbody>
    </table>
  );
}
