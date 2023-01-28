import Image from "../../../components/Image";
import { MynahICPoint } from "../../../utils/types";

export interface SelectedImageProps {
  selectedPoint: { pointIndex: number; pointClass: string } | null;
  data: Partial<Plotly.PlotData>[]
  selectedPointData: MynahICPoint | null
  mislabeledImages: string[]
}

export default function SelectedImage({
  selectedPoint, data, selectedPointData, mislabeledImages
}: SelectedImageProps): JSX.Element {
  if (selectedPointData === null || selectedPoint === null) return <div>Select a point or file in the list</div>;
  console.log('selectedimage render')
  const {x, y, fileid} = selectedPointData
  const { pointIndex, pointClass } = selectedPoint;
  const imgLoc = `/api/v1/file/${selectedPointData.fileid}/${selectedPointData.image_version_id}`
  const pointstr = `(${x.toFixed(2)}, ${y.toFixed(2)})`;
  return (
    <div className="flex h-full">
      <Image src={imgLoc} className="w-[60%] h-full mr-[15px]" />
      <ul>
        <li>Class: {pointClass}</li>
        <li>Mislabeled: {mislabeledImages.includes(fileid) ? 'yes' : 'no'}</li>
        <li>Point: {pointstr}</li>
        <li>PointIndex: {pointIndex}</li>
      </ul>
    </div>
  );
}
