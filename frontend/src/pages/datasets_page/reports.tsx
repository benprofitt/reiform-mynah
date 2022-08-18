import clsx from "clsx";
import _ from "lodash";
import { useState } from "react";
import { useQuery } from "react-query";
import { Link } from "wouter";
import makeRequest from "../../utils/apiFetch";
import getDate from "../../utils/date";
import {
  AsyncTaskData,
  MynahICDataset,
  MynahICReport,
} from "../../utils/types";
import DetailedReportView from "./detailed_report_view";

function ReportDropDown(props: {
  version: string;
  report: MynahICReport;
  basePath: string;
}) {
  const [open, setOpen] = useState(false);
  const { version, report, basePath } = props;
  const { data_id, date_created, tasks } = report;
  return (
    <div
      className={clsx(
        "border border-grey4 rounded-md text-[18px] h-fit mt-[10px] bg-white px-[15px] divide-y",
        !open && "cursor-pointer hover:shadow-floating"
      )}
    >
      <div
        className="w-full  flex items-center relative text-left h-[72px]"
        onClick={() => setOpen((open) => !open)}
      >
        <h6 className="w-[60%]">Version {version}</h6>
        <h6 className="w-[20%]">{getDate(date_created)}</h6>
        <h6 className="w-[20%]">{tasks.length}</h6>
        {/* {[version, data_id, date_created, tasks].toString(',')} */}
      </div>
      {open &&
        tasks.map((task, ix) => (
          <Link
            key={ix}
            className="hover:font-bold h-[55px] flex items-center cursor-pointer"
            to={`${basePath}/${data_id}/${task}`}
          >
            <h6>{task}</h6>
            <button className="ml-auto text-linkblue font-bold">View</button>
          </Link>
        ))}
    </div>
  );
}

export interface ReportsProps {
  dataset: MynahICDataset;
  pendingReports: string[] | null;
  basePath: string;
  reportId: string;
  reportType: string;
}

export default function Reports(props: ReportsProps) {
  const { dataset, pendingReports, basePath, reportId, reportType } = props;

  // temporary hard coded report
  const reports: MynahICDataset["reports"] = {
    "0": {
      data_id: "901dj012931iu3091",
      date_created: 12341212,
      tasks: ["ic::correct::class_splitting"],
    },
  };
  // i will show all completed tasks properly,
  // things that are pending and failed will show pending / failed
  // but will NOT show the task types as indicated in figma
  const {
    data: tasks,
    error,
    isLoading,
  } = useQuery<AsyncTaskData[]>("tasks", () =>
    makeRequest<AsyncTaskData[]>("GET", "/api/v1/task/list")
  );
  // const versionKeys = _.keys(dataset.reports);
  const versionKeysHard = ['0']
  const numCompleted = versionKeysHard.length;
  const completedPhrase = `${numCompleted} completed report${
    numCompleted !== 1 ? "s" : ""
  }`;
  if (error) return <div>error getting datasetFiles</div>;
  if (isLoading || !tasks)
    return (
      <div className="animate-spin aspect-square border-l border-r border-b border-sidebarSelected border-6 rounded-full w-[20px]" />
    );
  return (
    <div className="text-grey2 relative">
      <div className="flex">
        <h3>{completedPhrase}</h3>
      </div>
      <div className="mt-[30px] flex text-left">
        <h5 className="w-[60%]">Version</h5>
        <h5 className="w-[20%]">Date Created</h5>
        <h5 className="w-[20%]">Number of reports</h5>
      </div>
      {/* should be versionKeys (nohard) and dataset.reports */}
      {versionKeysHard.map((version, ix) => (
        <ReportDropDown
          key={ix}
          version={version}
          report={reports[version]}
          basePath={basePath}
        />
      ))}
      <DetailedReportView dataId={reportId} basePath={basePath} reportType={reportType} />
    </div>
  );
}
