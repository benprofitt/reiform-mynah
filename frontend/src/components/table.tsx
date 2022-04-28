import React from "react";

export interface TableProps {
  header: string[];
  data: {
    class: string;
    points: { x: number; y: number }[];
    bad: number;
    acceptable: number;
  }[];
  keys: string[];
}
export default function Table(props: TableProps): JSX.Element {
  const { header, data, keys } = props;
  return (
    <div className="flex flex-col">
      <div className="lg:-mx-4">
        <div className="inline-block min-w-full sm:px-6 lg:px-7">
          <table className="min-w-full items-center justify-center table-fixed border-collapse border border-black mx-auto">
            <thead>
              <tr>
                {header.map((head, index) => (
                  <th key={index} className="text-left">
                    {head}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr>
                {keys.map((key) => (
                  <td key={key} className="border border-black w-1/5">
                    {key}
                  </td>
                ))}
              </tr>
              {data.map((object, index) => (
                <tr key={index}>
                  <td className="border border-black py-1">
                    {object.class[5]}
                  </td>
                  <td className="border border-black">{object.bad}</td>
                  <td className="border border-black">{object.acceptable}</td>
                  <td className="border border-black">
                    {object.bad + object.acceptable}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
