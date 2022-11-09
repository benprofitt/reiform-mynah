import React from "react";

export interface ClassSplittingBreakdownProps {
  correctionReport: boolean;
}

export default function ClassSplittingBreakdown({
  correctionReport,
}: ClassSplittingBreakdownProps) {
  return (
    <table className="w-full text-[18px]">
      <th className="text-left text-[20px]">Class breakdown</th>
      <tr className="text-left"><td className="pt-[20px]">Cats</td></tr>
      <tr >
        <td className="pt-[20px]">Number of splits:</td>
        <td className="text-right">4</td>
      </tr>
      {correctionReport && (
        <tr className="text-left"><td className="pt-[20px]">New Class Names: Cats_0, Cats_1</td></tr>
      )}
      <td className="border-b-2 pt-[20px]" />
      <tr className="text-left"><td className="pt-[20px]">Dogs</td></tr>
      <tr>
        <td>Number of splits:</td>
        <td className="text-right">1</td>
      </tr>
      <td className="border-b-2 pt-[20px]" />
      <tr className="text-left"><td className="pt-[20px]">Birds</td></tr>
      <tr>
        <td className="pt-[20px]">Number of splits:</td>
        <td className="text-right">7</td>
      </tr>
      {correctionReport && (
        <tr className="text-left"><td className="pt-[20px]">New Class Names: Birds_0, Birds_1</td></tr>
      )}
      <td className="border-b-2 pt-[20px] w-full" />
    </table>
  );
}
