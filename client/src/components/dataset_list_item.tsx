import { MynahDataset } from "../types"

export interface DatasetListItemProps {
    dataset: MynahDataset
}

export default function DatasetListItem(props: DatasetListItemProps): JSX.Element {
    const {dataset} = props;
    const { dataset_name, dataset_type, date_modified } = dataset;
    return <li className="hover:shadow-floating cursor-pointer bg-white h-[70px] border border-grey4 rounded-md text-[18px] flex items-center mt-[10px] relative">
    <h6 className="ml-[10px] font-black text-black flex items-center">
      {/* <Image
        src={`/api/v1/file/${fileId}/latest`}
        className="h-full w-[70px] aspect-square mr-[1.5%] object-cover"
      /> */}
      {dataset_name}
    </h6>
    {/* <h6 className="w-[20%]">{version}</h6>
    <h6 className="w-[20%]">{getDate(date_created)}</h6>
    <h6 className="w-[20%]">{getDate(date_modified)}</h6> */}
    {/* <EllipsisMenu>
      <MenuItem src={OpenDatasetIcon} text="Open Dataset" />
      <MenuItem src={TrashIcon} text="Delete" />
    </EllipsisMenu> */}
  </li>
}