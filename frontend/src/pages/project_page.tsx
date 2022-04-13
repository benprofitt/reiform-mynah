import React, { useState } from "react";
import SideBar from "../components/sidebar";
import TopBar from "../components/topbar";
import CreateProject from "../project_tabs/create_project";
import DataDiagnosis from "../project_tabs/data_diagnosis";
import DiagnosisReport from "../project_tabs/diagnosis_report";
import DataCleaning from "../project_tabs/data_cleaning";
import CleaningReport from "../project_tabs/cleaning_report";
import DataEgress from "../project_tabs/data_egress";
import clsx from "clsx";
import { Switch, Route } from "wouter";
import SidebarTab from "../components/sidebar_tab";

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
  const { route } = props;
  const curFullPath = pathPrefix + route;
  const curRouteIndex = routes.indexOf(curFullPath);
  const isFinalTab = curRouteIndex === routes.length - 1;
  const nextRoutePath = !isFinalTab ? routes[curRouteIndex + 1] : curFullPath; // 'data-egress' since its the final tab;
  const isInvalidRoute = !routes.includes(curFullPath);

  const tabElements = [
    <CreateProject nextPath={nextRoutePath}/>,
    <DataDiagnosis />,
    <DiagnosisReport />,
    <DataCleaning />,
    <CleaningReport />,
    <DataEgress />,
  ];

  return (
    <>
      <TopBar>
        <h1 className="text-3xl ml-10 font-bold">{tabs[curRouteIndex]} Page</h1>
      </TopBar>
      <div className="flex grow">
        <SideBar>
          {tabs.map((tabName, index) => (
            <SidebarTab
              key={tabName}
              path={routes[index]}
              title={tabName}
              selected={curRouteIndex === index}
            />
          ))}
        </SideBar>
        <div className="w-full">
          <Switch>
            {tabElements.map((element, index) => (
              <Route key={index} path={routes[index]}>
                <div className="relative">{element}</div>
              </Route>
            ))}
          </Switch>
          {isInvalidRoute && (
            <div className="mt-3 text-center">This is not a valid path</div>
          )}
        </div>
      </div>
    </>
  );
}
