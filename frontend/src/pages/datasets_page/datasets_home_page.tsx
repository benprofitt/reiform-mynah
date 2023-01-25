import OpenDatasetIcon from "../../images/OpenDatasetIcon.svg";
import TrashIcon from "../../images/TrashIcon.svg";
import { EitherDataset } from "../../utils/types";
import { Link } from "wouter";
import { useQuery } from "react-query";
import makeRequest from "../../utils/apiFetch";
import Image from "../../components/Image";
import getDate from "../../utils/date";
import MenuItem from "../../components/menu_item";
import EllipsisMenu from "../../components/ellipsis_menu";
import FilterDropdown from "../../components/filter_dropdown";
import HomePageLayout from "../../components/home_page_layout";

interface DatasetListItemProps {
  dataset: EitherDataset;
}

const DatasetListItem = (props: DatasetListItemProps): JSX.Element => {
  const { dataset } = props;
  const { dataset_name, versions, date_created, date_modified, uuid } = dataset;
  const versionKeys = Object.keys(versions);
  const version = versionKeys.map((x) => parseInt(x)).at(-1);
  const someVersion = versions[versionKeys[0]];
  const fileId = Object.keys(someVersion["files"])[0];
  const isIC = "mean" in someVersion;
  return (
    <Link to={isIC ? `/datasets/ic/${uuid}` : `/datasets/od/${uuid}`}>
      <div className="hover:shadow-floating cursor-pointer bg-white h-[70px]border border-grey4 rounded-md text-[18px] flex items-center mt-[10px] relative">
        <h6 className="w-[40%] font-black text-black flex items-center">
          <Image
            src={`/api/v1/file/${fileId}/latest`}
            className="h-full w-[70px] aspect-square mr-[1.5%] object-cover"
          />
          {dataset_name}
        </h6>
        <h6 className="w-[20%]">{version}</h6>
        <h6 className="w-[20%]">{getDate(date_created)}</h6>
        <h6 className="w-[20%]">{getDate(date_modified)}</h6>
        <EllipsisMenu>
          <MenuItem src={OpenDatasetIcon} text="Open Dataset" />
          <MenuItem src={TrashIcon} text="Delete" />
        </EllipsisMenu>
      </div>
    </Link>
  );
};

interface MainDatasetContentProps {
  datasets: EitherDataset[];
}

const MainDatasetContent = (props: MainDatasetContentProps): JSX.Element => {
  const { datasets } = props;
  const numDatasets = datasets.length;
  const datasetCount =
    numDatasets === 1 ? "One Dataset" : `${numDatasets} total data sets`;
  return (
    <div className="text-grey2">
      <div className="flex justify-between">
        <h3>{datasetCount}</h3>
        <FilterDropdown />
      </div>
      <div className="mt-[30px] flex text-left">
        <h5 className="w-[40%]">Name</h5>
        <h5 className="w-[20%]">Version</h5>
        <h5 className="w-[20%]">Created</h5>
        <h5 className="w-[20%]">Last Modified</h5>
      </div>
      <div>
        {datasets.map((dataset, ix) => (
          <DatasetListItem dataset={dataset} key={ix} />
        ))}
      </div>
    </div>
  );
};

export default function DatasetsHomePage(): JSX.Element {
  const {
    isLoading,
    error,
    data: datasets,
  } = useQuery("datasets", () =>
    makeRequest<EitherDataset[]>("GET", "/api/v1/dataset/list")
  );

  console.log("home");

  if (error) return <div>error in datasets query</div>;

  const numDatasets = datasets?.length ?? 0;
  return (
    <HomePageLayout title="Datasets">
      {isLoading || datasets === undefined ? (
        <div className="rounded-full border-black w-[20px] aspect-square animate-spin" />
      ) : numDatasets === 0 ? (
        <h2>You have no datasets, add one by clicking the plus on the left</h2>
      ) : (
        <MainDatasetContent datasets={datasets} />
      )}
    </HomePageLayout>
  );
}
