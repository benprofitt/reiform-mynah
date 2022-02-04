import React, { useState } from "react";
import PageContainer from "../components/page_container";
import SideBar from "../components/sidebar";
import TopBar from "../components/topbar";
import DataIngest from "../project_tabs/data_ingest";
import DataDiagnosis from "../project_tabs/data_diagnosis";
import DiagnosisReport from "../project_tabs/data_ingest";
import DataCleaning from "../project_tabs/data_cleaning";
import CleaningReport from "../project_tabs/cleaning_report";
import DataEgress from "../project_tabs/data_egress";
import clsx from "clsx";

export interface ProjectPageProps {
  setIsProjectOpen: React.Dispatch<React.SetStateAction<boolean>>;
}

type TabName =
  | "Data Ingest"
  | "Data Diagnosis"
  | "Diagnosis Report"
  | "Data Cleaning"
  | "Cleaning Report"
  | "Data Egress";

export default function ProjectPage(props: ProjectPageProps): JSX.Element {
  const { setIsProjectOpen } = props;
  const [projectTitle, setProjectTitle] = useState("Template Project Title");
  const tabs: TabName[] = [
    "Data Ingest",
    "Data Diagnosis",
    "Diagnosis Report",
    "Data Cleaning",
    "Cleaning Report",
    "Data Egress",
  ];
  const TabMap: Record<TabName, JSX.Element> = {
    "Data Ingest": <DataIngest />,
    "Data Diagnosis": <DataDiagnosis />,
    "Diagnosis Report": <DiagnosisReport />,
    "Data Cleaning": <DataCleaning />,
    "Cleaning Report": <CleaningReport />,
    "Data Egress": <DataEgress />,
  };
  const [openTab, setOpenTab] = useState<TabName | null>(null);
  console.log(tabs[-1])
  return (
    <PageContainer>
      <TopBar onClickLogo={() => setIsProjectOpen(false)}>
        <div className="border-r-2 border-black w-full p-2 pr-5 flex">
          <input
            className="font-bold ml-10 text-4xl p-2 border border-black w-full"
            value={projectTitle}
            onChange={(e) => setProjectTitle(e.target.value)}
          />
        </div>
        <div className="flex items-center justify-center p-5 w-fit mx-auto">
          <button
            className={clsx(
              "border border-black w-36 h-full py-2",
              openTab === tabs[tabs.length - 1] && "pointer-events-none text-gray-300"
            )}
            onClick={() => {
              if (openTab === null) {
                setOpenTab(tabs[0]);
                return;
              }
              const nextTabNum = tabs.indexOf(openTab) + 1;
              if (nextTabNum >= tabs.length) {
                /* should never reach greater 
                than with pointer-events-none, 
                but why not be safe */
                setOpenTab(tabs[tabs.length - 1])
                return; 
              }
              setOpenTab(tabs[nextTabNum]);
            }}
          >
            Next
          </button>
        </div>
      </TopBar>
      <div className="flex grow">
        <SideBar>
          {tabs.map((tabName) => (
            <div
              key={tabName}
              className="border-t-2 border-b-2 border-black mb-5 p-2 cursor-pointer h-16"
              onClick={() => setOpenTab(tabName)}
            >
              {tabName}
            </div>
          ))}
        </SideBar>
        <div className="w-full">
          {openTab ? (
            TabMap[openTab]
          ) : (
            <p className="text-center mt-3">Project Contents</p>
          )}
        </div>
      </div>
    </PageContainer>
  );
}
