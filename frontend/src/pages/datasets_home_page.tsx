import clsx from "clsx";
import ArrowIcon from "../images/ArrowIcon.svg";
import OpenDatasetIcon from "../images/OpenDatasetIcon.svg";
import TrashIcon from "../images/TrashIcon.svg";
import { EitherDataset } from "../utils/types";
import { Menu } from "@headlessui/react";
import { Link } from "wouter";
import { useQuery } from "react-query";
import makeRequest from "../utils/apiFetch";
import Image from "../components/Image";
import getDate from "../utils/date";

const FilterDropdown = (): JSX.Element => {
  return (
    <Menu as="div" className="relative inline-block text-left ml-auto z-10">
      <div>
        <Menu.Button className="inline-flex w-[114px] h-[40px] items-center px-[10px] text-left rounded-md bg-white text-black border border-grey3 focus:outline-none">
          Filter by
          <img src={ArrowIcon} alt="arrow" className="ml-auto mt-[4px]" />
        </Menu.Button>
      </div>
      <Menu.Items className="absolute right-0 mt-2 w-56 origin-top-right divide-y divide-gray-100 rounded-md bg-white shadow-floating ring-1 ring-black ring-opacity-5 focus:outline-none">
        <DotMenuItem text="Date" />
        <DotMenuItem text="Size" />
      </Menu.Items>
    </Menu>
  );
};
interface DotMenuItemProps {
  src?: string;
  text: string;
}

const DotMenuItem = (props: DotMenuItemProps): JSX.Element => {
  const { src, text } = props;
  return (
    <Menu.Item>
      {({ active }) => (
        <button
          className={`${
            active ? "bg-sideBar text-white" : "text-gray-900"
          } flex w-full items-center rounded-md px-[20px] py-[15px]`}
        >
          {src && (
            <img
              src={src}
              alt="delete"
              className={clsx(
                "mr-[10px] h-[20px] aspect-square",
                active && "invert"
              )}
            />
          )}
          {text}
        </button>
      )}
    </Menu.Item>
  );
};

interface DatasetListItemProps {
  dataset: EitherDataset;
}

const DatasetListItem = (props: DatasetListItemProps): JSX.Element => {
  const { dataset } = props;
  const { dataset_name, versions, date_created, date_modified, uuid } = dataset;
  const versionKeys = Object.keys(versions);
  const version = versionKeys.map((x) => parseInt(x)).at(-1);
  const someVersion = versions[versionKeys[0]]
  const fileId = Object.keys(someVersion["files"])[0];
  const isIC = "mean" in someVersion
  return (
    <Link to={isIC ? `/dataset/ic/${uuid}` : `/dataset/od/${uuid}`}>
      <div className="hover:shadow-floating cursor-pointer bg-white h-fit border border-grey4 rounded-md text-[18px] flex items-center mt-[10px] relative">
        <h6 className="w-[40%] font-black text-black flex items-center">
          <Image
            src={`/api/v1/file/${fileId}/latest`}
            className="w-[9%] min-w-[70px] aspect-square mr-[1.5%] object-cover"
          />
          {dataset_name}
        </h6>
        <h6 className="w-[20%]">{version}</h6>
        <h6 className="w-[20%]">{getDate(date_created)}</h6>
        <h6 className="w-[20%]">{getDate(date_modified)}</h6>
        <Menu
          as="div"
          className="absolute inline-block text-left right-[15px] top-[30%]"
        >
          {({ open }) => (
            <>
              <div>
                <Menu.Button
                  className={clsx(
                    "hover:bg-grey3 transition-colors duration-300 rounded-full w-[30px] aspect-square flex items-center justify-center group",
                    open ? "bg-grey3" : "bg-clearGrey3"
                  )}
                >
                  {[0, 1, 2].map((ix) => (
                    <div
                      key={ix}
                      className={clsx(
                        "rounded-full w-[4px] aspect-square mx-[2px] transition-colors duration-300 group-hover:bg-grey5 ",
                        open ? "bg-grey5" : "bg-grey6"
                      )}
                    />
                  ))}
                </Menu.Button>
              </div>
              <Menu.Items className="z-10 absolute right-[15px] mt-[15px] w-56 origin-top-right rounded-md bg-white shadow-floating focus:outline-none">
                <DotMenuItem src={OpenDatasetIcon} text="Open Dataset" />
                <DotMenuItem src={TrashIcon} text="Delete" />
              </Menu.Items>
            </>
          )}
        </Menu>
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
      <div className="flex">
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

  if (error) return <div>error in datasets query</div>;

  const numDatasets = datasets?.length ?? 0;
  return (
    <div className="flex h-screen flex-1">
      <div className="w-full flex flex-col h-full">
        <header className="w-full h-[80px] border-b border-grey1 pl-[32px] flex items-center bg-white">
          <h1 className="font-black text-[28px]">Datasets</h1>
        </header>
        <main className="bg-grey w-full p-[32px] flex-1 overflow-y-scroll">
          {isLoading || datasets === undefined ? (
            <div className="rounded-full border-black w-[20px] aspect-square animate-spin" />
          ) : numDatasets === 0 ? (
            <h2>
              You have no datasets, add one by clicking the plus on the left
            </h2>
          ) : (
            <MainDatasetContent datasets={datasets} />
          )}
        </main>
      </div>
    </div>
  );
}
