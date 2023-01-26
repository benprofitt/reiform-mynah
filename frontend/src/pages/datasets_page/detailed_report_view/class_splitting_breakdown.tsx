import React from "react";
import { MynahICProcessTaskDiagnoseClassSplittingReport } from "../../../utils/types";

export interface ClassSplittingBreakdownProps {
  correctionReport: boolean;
  taskMetadata: MynahICProcessTaskDiagnoseClassSplittingReport;
}

export default function ClassSplittingBreakdown({
  correctionReport,
  taskMetadata,
}: ClassSplittingBreakdownProps) {
  return (
    <table className="w-full text-[18px]">
      <thead>
        <tr>
          <th className="text-left text-[20px]">Class breakdown</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(taskMetadata.classes_splitting).map(
          ([className, { predicted_classes_count }]) => (
            <>
              <tr className="text-left">
                <td className="pt-[20px]">Class {className}</td>
              </tr>
              <tr>
                <td className="pt-[20px]">Number of splits:</td>
                <td className="text-right">{predicted_classes_count}</td>
              </tr>
              <tr className="border-b-2" />

              {/*  TODO GOTTA HANDLE CORRECTION
        {correctionReport && (
          <tr className="text-left">
            <td className="pt-[20px]">New Class Names: Cats_0, Cats_1</td>
          </tr>
        )} */}
            </>
          )
        )}
        <tr className="border-b-2" />
      </tbody>
    </table>
  );
}
