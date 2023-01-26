import Image from "../../../components/Image";
import { MynahICPoint } from "../../../utils/types";

export interface SelectedImageProps {
  last: { x: Plotly.Datum; y: Plotly.Datum; pointIndex: number; pointClass: number } | null;
  data: Partial<Plotly.PlotData>[]
  selectedPointData: MynahICPoint | null
}

export default function SelectedImage({
  last, data, selectedPointData
}: SelectedImageProps): JSX.Element {
  if (selectedPointData === null || last === null) return <div>Select a point or file in the list</div>;
  console.log('selectedimage render')
  const { x, y, pointIndex, pointClass } = last;
  const imgLoc = `/api/v1/file/${selectedPointData.fileid}/${selectedPointData.image_version_id}`
  const pointstr = `(${Number(x).toFixed(2)}, ${Number(y).toFixed(2)})`;
  return (
    <div className="flex h-full">
      <Image src={imgLoc} className="w-[60%] h-full mr-[15px]" />
      <ul>
        <li>Class: {pointClass}</li>
        <li>Mislabeled: yes</li>
        <li>Point: {pointstr}</li>
        <li>PointIndex: {pointIndex}</li>
      </ul>
    </div>
  );
}
