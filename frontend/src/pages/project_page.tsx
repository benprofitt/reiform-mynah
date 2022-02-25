import React, { useState } from "react";
import PageContainer from "../components/page_container";
import SideBar from "../components/sidebar";
import TopBar from "../components/topbar";
import DataIngest from "../project_tabs/data_ingest";
import DataDiagnosis from "../project_tabs/data_diagnosis";
import DiagnosisReport from "../project_tabs/diagnosis_report";
import DataCleaning from "../project_tabs/data_cleaning";
import CleaningReport from "../project_tabs/cleaning_report";
import DataEgress from "../project_tabs/data_egress";
import clsx from "clsx";
import { Routes, Route, Link, useLocation } from "react-router-dom";

type TabName =
  | "Data Ingest"
  | "Data Diagnosis"
  | "Diagnosis Report"
  | "Data Cleaning"
  | "Cleaning Report"
  | "Data Egress";

function toRouteName(tabName: TabName): string {
  return tabName.toLocaleLowerCase().replace(" ", "-");
}

const tabs: TabName[] = [
  "Data Ingest",
  "Data Diagnosis",
  "Diagnosis Report",
  "Data Cleaning",
  "Cleaning Report",
  "Data Egress",
];

const tabElements: JSX.Element[] = [
  <DataIngest />,
  <DataDiagnosis />,
  <DiagnosisReport />,
  <DataCleaning />,
  <CleaningReport />,
  <DataEgress />,
];

const routes: string[] = tabs.map((tabName) => toRouteName(tabName));
// [data-ingest, data-diagnosis, ...]

const pathPrefix = "/mynah/project/";

export default function ProjectPage(): JSX.Element {
  const [projectTitle, setProjectTitle] = useState("Template Project Title");

  const { pathname } = useLocation();
  // '/mynah/project/data-ingest' => 'data-ingest'
  const curRouteName = pathname.slice(pathPrefix.length);
  const curRouteIndex = routes.indexOf(curRouteName);
  const isFinalTab = curRouteIndex === routes.length - 1;
  const nextRoutePath = !isFinalTab
    ? routes[curRouteIndex + 1]
    : curRouteName // 'data-egress' since its the final tab;
  const isInvalidRoute = !routes.includes(curRouteName);

  return (
    <PageContainer>
      <TopBar>
        <div className="border-r-2 border-black w-full p-2 pr-5 flex">
          <input
            className="font-bold ml-10 text-4xl p-2 border border-black w-full"
            value={projectTitle}
            onChange={(e) => setProjectTitle(e.target.value)}
          />
        </div>
        <div className="flex items-center justify-center p-5 w-fit mx-auto">
          <Link
            className={clsx(
              "border text-center text-black border-black w-36 h-full py-2",
              isFinalTab && "pointer-events-none text-gray-500"
            )}
            to={nextRoutePath}
          >
            Next
          </Link>
        </div>
      </TopBar>
      <div className="flex grow">
        <SideBar>
          {tabs.map((tabName, index) => (
            <div
              key={tabName}
              className={clsx(
                "border-t-2 border-b-2 border-black mb-5 p-2 cursor-pointer h-16",
                curRouteName === routes[index] && "font-bold"
              )}
            >
              <Link to={routes[index]}>{tabName}</Link>
            </div>
          ))}
        </SideBar>
        <div className="w-full">
          <Routes>
            {tabElements.map((element, index) => (
              <Route key={index} path={routes[index]} element={element} />
            ))}
          </Routes>
          {isInvalidRoute && (
            <div className="mt-3 text-center">This is not a valid path</div>
          )}
        </div>
      </div>
    </PageContainer>
  );
}
