import Ellipsis from "../../../images/Ellipsis.svg";
import { Redirect, RouteComponentProps, useLocation } from "wouter";
import { Tab } from "@headlessui/react";
import { MynahICDataset, MynahICProcessTaskType } from "../../../utils/types";
import { useQuery } from "react-query";
import makeRequest from "../../../utils/apiFetch";
import { useState } from "react";
import DetailPageWithTabs from "../../../components/detail_page_with_tabs";
import Configuration from "./configuration";
import Results from "./results";

const tabNames = ["configuration", "results"];

export default function ModelDetailPage(
  props: RouteComponentProps<{
    uuid: string;
    tab: string;
  }>
): JSX.Element {
  const { uuid, tab } = props.params;
  const basePath = `/models/${uuid}`;

  if (!tabNames.includes(tab))
    return <Redirect to={`${basePath}/configuration`} />;

  const selectedIndex = tab === "results" ? 1 : 0;

  return (
    <DetailPageWithTabs
      title="Model Name"
      backButtonDestination="/models"
      basePath={basePath}
      selectedIndex={selectedIndex}
      topRightButtons={
        <>
          <button
            className="bg-linkblue text-white h-full w-[134px] text-center font-medium rounded-md"
            onClick={() => console.log("click")}
          >
            Train Model
          </button>
          {/* <ProcessDataModal
            isOpen={processDataOpen}
            close={() => setProcessDataOpen(false)}
            uuid={uuid}
            setPendingReports={setPendingReports}
          /> */}
          <button className="h-full">
            <img src={Ellipsis} alt="more" />
          </button>
        </>
      }
      tabNames={tabNames}
      tabPanels={
        <Tab.Panels className='h-full'>
          <Tab.Panel>
            <Configuration />
          </Tab.Panel>
          <Tab.Panel className='h-full'>
            <Results />
          </Tab.Panel>
        </Tab.Panels>
      }
    />
  );
}
