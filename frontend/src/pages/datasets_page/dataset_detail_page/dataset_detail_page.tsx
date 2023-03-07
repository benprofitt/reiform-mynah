import Ellipsis from "../../../images/Ellipsis.svg";
import { Redirect, RouteComponentProps, useLocation } from "wouter";
import { Tab } from "@headlessui/react";
import { MynahICDataset, MynahICProcessTaskType } from "../../../utils/types";
import { useQuery } from "react-query";
import makeRequest from "../../../utils/apiFetch";
import { useState } from "react";
import Files from "./files";
import Reports from "./reports";
import ProcessDataModal from "./process_data_modal";
import DetailPageWithTabs from "../../../components/detail_page_with_tabs";

const tabNames = ["files", "reports"];

export default function DatasetDetailPage(
  props: RouteComponentProps<{
    uuid: string;
    tab: string;
    id: string;
    type: string;
  }>
): JSX.Element {
  const [_location, setLocation] = useLocation();
  const { uuid, tab, id, type } = props.params;
  const { isLoading, data: dataset } = useQuery<MynahICDataset>(
    `dataset-${uuid}`,
    () => makeRequest<MynahICDataset>("GET", `/api/v1/dataset/ic/${uuid}`)
  );
  const basePath = `/datasets/ic/${uuid}`;
  console.log({ uuid, tab, id, type });
  const [processDataOpen, setProcessDataOpen] = useState(false);

  if (!tabNames.includes(tab)) return <Redirect to={`${basePath}/files`} />;

  const selectedIndex = tab === "reports" ? 1 : 0;

  if (isLoading)
    return (
      <div className="rounded-full border-black w-[20px] aspect-square animate-spin" />
    );
  if (dataset === undefined)
    return <div>Unable to retrive datasets, are you logged in?</div>;

  const { dataset_name: name } = dataset;

  return (
    <DetailPageWithTabs
      title={name}
      basePath={basePath}
      backButtonDestination="/"
      selectedIndex={selectedIndex}
      topRightButtons={
        <>
          <button className="text-linkblue h-full w-[134px] text-center font-medium">
            Add More Data
          </button>
          <button
            className="bg-linkblue text-white h-full w-[134px] text-center font-medium rounded-md"
            onClick={() => setProcessDataOpen(true)}
          >
            Process Data
          </button>
          <ProcessDataModal
            isOpen={processDataOpen}
            close={() => setProcessDataOpen(false)}
            uuid={uuid}
          />
          <button className="h-full">
            <img src={Ellipsis} alt="more" />
          </button>
        </>
      }
      tabNames={tabNames}
      tabPanels={
        <Tab.Panels as='div' className='h-full'>
          <Tab.Panel as='div' className='h-full'>
            <Files
              dataset={dataset}
              basePath={`${basePath}/files`}
              fileId={id}
            />
          </Tab.Panel>
          <Tab.Panel>
            <Reports
              dataset={dataset}
              basePath={`${basePath}/reports`}
              reportId={id}
              reportType={type as MynahICProcessTaskType}
            />
          </Tab.Panel>
        </Tab.Panels>
      }
    />
  );
}
