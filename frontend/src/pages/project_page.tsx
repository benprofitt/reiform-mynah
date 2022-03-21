import React, { useState } from "react";
import PageContainer from "../components/page_container";
import SideBar from "../components/sidebar";
import TopBar from "../components/topbar";
import CreateProject from "../project_tabs/create_project";
import DataDiagnosis from "../project_tabs/data_diagnosis";
import DiagnosisReport from "../project_tabs/diagnosis_report";
import DataCleaning from "../project_tabs/data_cleaning";
import CleaningReport from "../project_tabs/cleaning_report";
import DataEgress from "../project_tabs/data_egress";
import clsx from "clsx";
import { Switch, Route, Link, RouteComponentProps } from "wouter";

type TabName =
  | "Create Project"
  | "Data Diagnosis"
  | "Diagnosis Report"
  | "Data Cleaning"
  | "Cleaning Report"
  | "Data Egress";

const tabs: TabName[] = [
  "Create Project",
  "Data Diagnosis",
  "Diagnosis Report",
  "Data Cleaning",
  "Cleaning Report",
  "Data Egress",
];

const tabElements: React.ComponentType<
  RouteComponentProps<{
    [x: string]: string;
  }>
>[] = [
  CreateProject,
  DataDiagnosis,
  DiagnosisReport,
  DataCleaning,
  CleaningReport,
  DataEgress,
];


const pathPrefix = "/project/";

function toRouteName(tabName: TabName): string {
  return pathPrefix + tabName.toLocaleLowerCase().replace(" ", "-");
}

const routes: string[] = tabs.map((tabName) => toRouteName(tabName));
// [data-ingest, data-diagnosis, ...]


export interface ProjectPageProps {
  route: string;
}

export default function ProjectPage(props: ProjectPageProps): JSX.Element {
  const [projectTitle, setProjectTitle] = useState("Template Project Title");

  const { route } = props;
  const curFullPath = pathPrefix + route
  const curRouteIndex = routes.indexOf(curFullPath);
  const isFinalTab = curRouteIndex === routes.length - 1;
  const nextRoutePath = !isFinalTab ? routes[curRouteIndex + 1] : curFullPath; // 'data-egress' since its the final tab;
  const isInvalidRoute = !routes.includes(curFullPath);

  console.log(curRouteIndex, nextRoutePath)

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
                curFullPath === routes[index] && "font-bold"
              )}
            >
              <Link to={routes[index]}>{tabName}</Link>
            </div>
          ))}
        </SideBar>
        <div className="w-full">
          <Switch>
            {tabElements.map((element, index) => (
              <Route key={index} path={routes[index]} component={element} />
            ))}
          </Switch>
          {isInvalidRoute && (
            <div className="mt-3 text-center">This is not a valid path</div>
          )}
        </div>
      </div>
    </PageContainer>
  );
}
