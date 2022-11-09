export interface SelectedImageProps {
  last: { x: Plotly.Datum; y: Plotly.Datum; pointIndex: number } | null;
}

export default function SelectedImage({
  last,
}: SelectedImageProps): JSX.Element {
  if (last === null) return <div>select a point or file in the list</div>;
  console.log('selectedimage render')
  const { x, y, pointIndex } = last;
  const pointstr = `(${Number(x).toFixed(2)}, ${Number(y).toFixed(2)})`;
  return (
    <div className="flex h-full">
      <div className="w-[60%] h-full bg-gray-500 mr-[15px]" />
      <ul>
        <li>Class: 4</li>
        <li>Mislabeled: yes</li>
        <li>Point: {pointstr}</li>
        <li>PointIndex: {pointIndex}</li>
      </ul>
    </div>
  );
}
