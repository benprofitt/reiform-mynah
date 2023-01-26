import clsx from "clsx";
import _ from "lodash";
import { useState } from "react";
import { useQuery } from "react-query";
import { Link } from "wouter";
import makeRequest from "../../../utils/apiFetch";
import getDate from "../../../utils/date";
import {
  AsyncTaskData,
  MynahICDataset,
  MynahICProcessTaskType,
  MynahICReport,
} from "../../../utils/types";
import DetailedReportView from "../detailed_report_view/detailed_report_view";
import ArrowIcon from "../../../images/ArrowIcon.svg";
import ReportIcon from "../../../images/ReportIcon.svg";

export const reportToString: Record<MynahICProcessTaskType, string> = {
  "ic::diagnose::mislabeled_images": "Labeling Diagnosis Report",
  "ic::correct::mislabeled_images": "Labeling Correction Report",
  "ic::diagnose::class_splitting": "Variance Diagnosis Report",
  "ic::correct::class_splitting": "Variance Correction Report",
};

function PendingReport(props: { version: number; date_created: number }) {
  const { version, date_created } = props;
  return (
    <div className="w-full hover:shadow-floating cursor-pointer bg-white border border-grey4 rounded-md text-[18px] flex items-center mt-[10px] relative text-left h-[72px] px-[15px]">
      <h6 className="w-[60%] flex items-center">
        <span className="ml-[10px] text-grey6">(Currently Processing...)</span> <div className="ml-[10px] animate-spin aspect-square border-l border-r border-b border-red-500 rounded-full w-[20px]" />
      </h6>
      <h6 className="w-[20%]">{getDate(date_created)}</h6>
    </div>
  );
}

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
        <img
          src={ArrowIcon}
          alt="arrow"
          className={clsx("absolute right-[4px]", open && "rotate-180")}
        />
        {/* {[version, data_id, date_created, tasks].toString(',')} */}
      </div>
      {open &&
        tasks.map((task, ix) => (
          <Link
            key={ix}
            className="hover:font-bold h-[55px] flex items-center cursor-pointer"
            to={`${basePath}/${data_id}/${task}`}
          >
            <img src={ReportIcon} className="mr-[10px]" />
            <h6>{reportToString[task]}</h6>
            <button className="ml-auto text-linkblue font-bold">View</button>
          </Link>
        ))}
    </div>
  );
}

export interface ReportsProps {
  dataset: MynahICDataset;
  basePath: string;
  reportId: string;
  reportType: MynahICProcessTaskType;
}

export default function Reports(props: ReportsProps) {
  const { dataset, basePath, reportId, reportType } = props;

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
  
  const pendingReports = tasks?.filter(
    (task) => ['running', 'pending'].includes(task.task_status)
  );
  const versionKeys = _.keys(dataset.reports);
  const maxVersion = Math.max(...versionKeys.map((x) => parseInt(x)));
  const numCompleted = versionKeys.length;
  const completedPhrase = `${numCompleted} completed report${
    numCompleted !== 1 ? "s" : ""
  }`;
  const curSelectedVersion = Object.entries(dataset.reports).filter(
    ([, { data_id }]) => data_id === reportId
  )[0]?.[0];
  if (error) return <div>error getting datasetFiles</div>;
  if (isLoading || !tasks)
    return (
      <div className="animate-spin aspect-square border-l border-r border-b border-sidebarSelected border-6 rounded-full w-[20px]" />
    );
  return Boolean(reportId) ? (
    <DetailedReportView
      dataId={reportId}
      dataset={dataset}
      basePath={basePath}
      reportType={reportType}
      version={curSelectedVersion}
    />
  ) : (
    <div className="text-grey2">
      <div className="flex">
        <h3>{completedPhrase}</h3>
      </div>
      <div className="mt-[30px] flex text-left">
        <h5 className="w-[60%]">Version</h5>
        <h5 className="w-[20%]">Date Created</h5>
        <h5 className="w-[20%]">Number of reports</h5>
      </div>
      {versionKeys.map((version, ix) => (
        <ReportDropDown
          key={ix}
          version={version}
          report={dataset.reports[version]}
          basePath={basePath}
        />
      ))}
      {pendingReports?.map((x) => (
        <PendingReport key={x.task_id} date_created={x.started} version={maxVersion + 1} />
      ))}
    </div>
  );
}
