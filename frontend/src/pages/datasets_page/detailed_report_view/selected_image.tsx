import Image from "../../../components/Image";
import { MynahICPoint } from "../../../utils/types";
import { MislabeledType } from "./image_list_viewer_and_scatter";

export interface SelectedImageProps {
  selectedPoint: { pointIndex: number; pointClass: string } | null;
  data: Partial<Plotly.PlotData>[]
  selectedPointData: MynahICPoint | null
  getMislabeledType: (fileId: string, className: string) => MislabeledType
}

const mislabeledTypeToString: Record<MislabeledType, string> = {
  'mislabeled': 'Yes',
  'unchanged': 'No',
  'mislabeled_corrected': 'Yes. Corrected',
  'mislabeled_removed': 'Yes. Removed'
}

export default function SelectedImage({
  selectedPoint, data, selectedPointData, getMislabeledType
}: SelectedImageProps): JSX.Element {
  if (selectedPointData === null || selectedPoint === null) return <div>Select a point or file in the list</div>;
  console.log('selectedimage render')
  const {x, y, fileid, original_class} = selectedPointData
  const { pointClass } = selectedPoint;
  const imgLoc = `/api/v1/file/${selectedPointData.fileid}/${selectedPointData.image_version_id}`
  const pointstr = `(${x.toFixed(2)}, ${y.toFixed(2)})`;
  const mislabeledType = getMislabeledType(fileid, pointClass)
  return (
    <div className="flex h-full">
      <Image src={imgLoc} className="w-[60%] h-full mr-[15px]" />
      <ul>
        <li><b>Current Class:</b> {pointClass}</li>
        <li>Original Class: {original_class}</li>
        <li>Point: {pointstr}</li>
        <li>Potentially Mislabled: {mislabeledTypeToString[mislabeledType]}</li>
      </ul>
    </div>
  );
}
