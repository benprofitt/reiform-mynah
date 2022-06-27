import Ellipsis from "../images/Ellipsis.svg";
import BackArrowIcon from "../images/BackArrowIcon.svg";
import { Link, RouteComponentProps } from "wouter";
import { Menu, Tab } from "@headlessui/react";
import clsx from "clsx";
import {
  EitherData,
  EitherDataset,
  EitherFile,
  MynahFile,
} from "../utils/types";
import { useQuery } from "react-query";
import makeRequest from "../utils/apiFetch";
import Image from "../components/Image";

const MyTab = (props: { text: string }): JSX.Element => {
  const { text } = props;
  return (
    <Tab>
      {({ selected }) => (
        <div
          className={clsx(
            "relative h-[40px] mr-[20px] mt-[20px] font-bold uppercase",
            selected ? "text-black" : "text-grey2"
          )}
        >
          {text}
          {selected && (
            <div className="absolute bottom-0 w-full bg-linkblue h-[5px] rounded-sm"></div>
          )}
        </div>
      )}
    </Tab>
  );
};

const File = (props: {
  version: string;
  fileId: string;
  file: EitherFile;
  data: MynahFile;
}): JSX.Element => {
  const { version, fileId, file } = props;

  return (
    <div className="hover:shadow-floating cursor-pointer bg-white h-fit border border-grey4 rounded-md text-[18px] flex items-center mt-[10px] relative">
      <h6 className="w-[40%] font-black text-black flex items-center">
        <Image
          src={`/api/v1/file/${fileId}/${file["image_version_id"]}`}
          className="w-[9%] min-w-[70px] aspect-square mr-[1.5%] object-cover"
        />
        {fileId}
      </h6>
      <h6 className="w-[20%]">{version}</h6>
      {/* <h6 className="w-[20%]">{dateCreated}</h6>
  <h6 className="w-[20%]">{dateModified}</h6> */}
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
              {/* <DotMenuItem src={OpenDatasetIcon} text="Open Dataset" />
          <DotMenuItem src={TrashIcon} text="Delete" /> */}
            </Menu.Items>
          </>
        )}
      </Menu>
    </div>
  );
};

const Files = (props: { dataset: EitherDataset }): JSX.Element => {
  const { dataset } = props;
  const files = Object.entries(dataset.versions)
    .reverse()
    .map(([version, data]: [string, EitherData]) =>
      Object.entries(data.files).map(
        ([fileId, file]: [string, EitherFile]) => ({
          version,
          fileId,
          file,
        })
      )
    )
    .flat();
  const numFiles = files.length;
  const fileCount = numFiles === 1 ? "1 photo" : `${numFiles} photos`;

  const ids = files.map(({ fileId }) => fileId);
  const query = `/api/v1/file/list?fileid=${ids.join("&fileid=")}`;
  const { error, isLoading, data } = useQuery("datasetFiles", () =>
    makeRequest<Record<string, MynahFile>>("GET", query)
  );

  if (error) return <div>error getting datasetFiles</div>;
  if (isLoading)
    return (
      <div className="animate-spin aspect-square border-l border-r border-b border-sidebarSelected border-6 rounded-full w-[20px]" />
    );
  return (
    <div className="text-grey2">
      <div className="flex">
        <h3>
          {fileCount} {query}
        </h3>
      </div>
      <div className="mt-[30px] flex text-left">
        <h5 className="w-[40%]">Name</h5>
        <h5 className="w-[20%]">Date</h5>
        <h5 className="w-[20%]">Classes</h5>
        <h5 className="w-[20%]">Version</h5>
      </div>
      {data &&
        files.map((file, ix) => (
          <File key={ix} {...file} data={data[file.fileId]} />
        ))}
    </div>
  );
};

export default function DatasetPage(
  props: RouteComponentProps<{ uuid: string }>
): JSX.Element {
  const { data: datasets } = useQuery<EitherDataset[]>("datasets", () =>
    makeRequest<EitherDataset[]>("GET", "/api/v1/dataset/list")
  );
  const { uuid } = props.params;
  console.log("got the uuid", uuid);
  if (datasets === undefined)
    return <div>Unable to retrive datasets, are you logged in?</div>;
  const dataset = datasets.find((dataset) => dataset.uuid === uuid);
  if (dataset === undefined)
    return (
      <div>
        This dataset either does not exist or you do not have permission to see
        it
      </div>
    );
  const { dataset_name: name } = dataset;
  return (
    <div className="flex h-screen flex-1">
      <Tab.Group as="div" className="w-full flex flex-col">
        <header className="w-full h-fit border-b border-grey1 pl-[32px] relative bg-white pt-[46px]">
          <h1 className="font-bold text-[28px]">{name}</h1>
          <Link to="/">
            <button className="flex items-center text-linkblue absolute left-[30px] top-[20px] font-bold">
              <img src={BackArrowIcon} alt="back arrow" className="mr-[2px]" />
              Back
            </button>
          </Link>
          <div className="absolute right-[36px] top-[60px] space-x-[20px] text-[18px] h-[38px] flex items-center">
            <button className="text-linkblue h-full w-[134px] text-center font-medium">
              Add More Data
            </button>
            <button className="bg-linkblue text-white h-full w-[134px] text-center font-medium rounded-md">
              Process Data
            </button>
            <button className="h-full">
              <img src={Ellipsis} alt="more" />
            </button>
          </div>
          <Tab.List>
            <MyTab text="Files" />
            <MyTab text="Diagnosis" />
            <MyTab text="Cleaning" />
          </Tab.List>
        </header>
        <main className="bg-grey w-full p-[32px] flex-1">
          <Tab.Panels>
            <Tab.Panel>
              <Files dataset={dataset} />
            </Tab.Panel>
            <Tab.Panel>Diagnosis Tab</Tab.Panel>
            <Tab.Panel>Cleaning Tab</Tab.Panel>
          </Tab.Panels>
        </main>
      </Tab.Group>
    </div>
  );
}
