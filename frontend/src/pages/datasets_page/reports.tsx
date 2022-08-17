import clsx from "clsx";
import { useState } from "react";
import { useQuery } from "react-query";
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
  onClick: () => void;
}) {
  const [open, setOpen] = useState(false);
  const { version, report, onClick } = props;
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
          <div
            key={ix}
            className="hover:font-bold h-[55px] flex items-center cursor-pointer"
            onClick={onClick}
          >
            <h6>{task}</h6>
            <button className="ml-auto text-linkblue font-bold">View</button>
          </div>
        ))}
    </div>
  );
}

export interface ReportsProps {
  dataset: MynahICDataset;
  pendingReports: string[] | null;
}

export default function Reports(props: ReportsProps) {
  const { dataset, pendingReports } = props;

  // instead of this should just be dataset.reports
  const reports: MynahICDataset["reports"] = {
    "0": {
      data_id: "901dj012931iu3091",
      date_created: 12341212,
      tasks: ["ic::correct::class_splitting"],
    },
  };
  // i can get all the completed tasks this way, but to get the uncompleted ones
  // problem here, async task data does not include the task type but it looks in figma like that is desired.
  const {
    data: tasks,
    error,
    isLoading,
  } = useQuery<AsyncTaskData[]>("tasks", () =>
    makeRequest<AsyncTaskData[]>("GET", "/api/v1/task/list")
  );
  const [selectedReport, setSelectedReport] = useState<number | null>(null);
  const theReports = Object.entries(reports);
  const curDataId =
    selectedReport === null ? null : theReports[selectedReport][1].data_id;
  const numCompleted = theReports.length;
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
      {theReports.map(([version, report], ix) => (
        <ReportDropDown
          key={ix}
          version={version}
          report={report}
          onClick={() => setSelectedReport(ix)}
        />
      ))}
      {curDataId !== null && (
        <DetailedReportView
          dataId={curDataId}
          close={() => setSelectedReport(null)}
        />
      )}
    </div>
  );
}
