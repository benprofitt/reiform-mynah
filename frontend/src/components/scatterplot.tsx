import React from "react";
import {
  Scatter,
  ScatterChart,
  XAxis,
  Tooltip,
  YAxis,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
} from "recharts";
export interface ScatterplotProps {
  Yaxis: string;
  Xaxis: string;
  data: {
    class: string;
    points: { x: number; y: number }[];
    bad: number;
    acceptable: number;
  }[];
  colors: string[];
}

export default function Scatterplot(props: ScatterplotProps): JSX.Element {
  const { Yaxis, Xaxis, data, colors } = props;
  return (
    <ResponsiveContainer width="90%" height="40%">
      <ScatterChart margin={{ top: 10, right: 20, bottom: 15, left: 20 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" dataKey={Xaxis} />
        <YAxis dataKey={Yaxis} />
        <Tooltip />
        <Legend
          verticalAlign="bottom"
          height={10}
          align="right"
          iconSize={10}
        />
        {data.map((_class, index) => (
          <Scatter
            key={_class.class}
            name={_class.class}
            data={_class.points}
            fill={colors[index]}
            onClick={(e) => console.log(e.node)}
          />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  );
}
