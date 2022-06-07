import React from "react";
// import {
//   BarChart,
//   Bar,
//   XAxis,
//   Tooltip,
//   YAxis,
//   CartesianGrid,
//   Legend,
//   ResponsiveContainer,
// } from "recharts";

export interface HistogramProps {
  Xaxis: string;
  data: {
    class: string;
    points: { x: number; y: number }[];
    bad: number;
    acceptable: number;
  }[];
}
export default function Histogram(props: HistogramProps): JSX.Element {
  const { data, Xaxis } = props;
  return (
    // <ResponsiveContainer width="90%" height="42%">
    //   <BarChart
    //     data={data}
    //     margin={{ top: 0, right: 10, left: 20, bottom: 10 }}
    //   >
    //     <CartesianGrid strokeDasharray="3 3" />
    //     <XAxis dataKey={Xaxis} />
    //     <YAxis />
    //     <Tooltip />
    //     <Legend
    //       verticalAlign="top"
    //       width={200}
    //       height={23}
    //       align="right"
    //       iconSize={10}
    //     />
    //     <Bar dataKey="acceptable" fill="#8884d8" />
    //     <Bar dataKey="bad" fill="#82ca9d" />
    //   </BarChart>
    // </ResponsiveContainer>
    <></>
  );
}
