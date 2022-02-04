import clsx from "clsx";
import React, { useState } from "react";
import PageContainer from "./../components/page_container";
import TopBar from "../components/topbar";
import SideBar from "../components/sidebar";

export interface HomePageProps {
  setIsProjectOpen: React.Dispatch<React.SetStateAction<boolean>>;
}

type Options = "Projects" | "Datasets";

export default function HomePage(props: HomePageProps): JSX.Element {
  const { setIsProjectOpen } = props;
  const [mode, setMode] = useState<Options>("Projects");
  const options: Options[] = ["Projects", "Datasets"];
  const headers = ["Name", "Datasets", "Date Created", "Download"];

  return (
    <PageContainer>
      <TopBar>
        <div className="border-r-2 border-black w-full p-2 flex">
          {options.map((options) => {
            return (
              <div
                className={clsx(
                  "h-full w-1/3 border border-black ml-10 font-bold pt-4 pl-4",
                  mode === options
                    ? "bg-blue-300 pointer-events-none"
                    : "cursor-pointer"
                )}
                onClick={() => setMode(options)}
                key={options}
              >
                {options}
              </div>
            );
          })}
        </div>
        <div className="flex items-center justify-center w-fit p-5 mx-auto">
          <button
            className="border border-black w-36 h-full py-2"
            onClick={() => {
              if (mode === 'Datasets') return
              setIsProjectOpen(true)}}
          >
            + New {mode.slice(0, -1)}
          </button>
        </div>
      </TopBar>
      <div className="w-full grow flex relative">
        <SideBar>
          <div className="p-2 hidden">Home page Sidebar stuff</div>
        </SideBar>
        <div className="w-full absolute">
          {[0, 1, 2, 3, 4, 5].map((ix) => (
            <div
              key={ix}
              className={clsx(
                "w-full h-20 border-b-2 border-black flex items-center",
                ix === 0 ? "h-16" : "h-24"
              )}
            >
              <div className="ml-[35px] w-6 h-6 shrink-0 border-2 border-black" />
              {ix === 0 && (
                <div className="ml-24 mr-10 flex justify-between w-full">
                  {headers.map((header) => (
                    <div className="w-56" key={header}>
                      <div
                        className={clsx(
                          "border-2 border-black text-center",
                          mode === "Datasets" &&
                            ["Datasets", "Download"].includes(header) &&
                            "hidden"
                        )}
                      >
                        {header}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </PageContainer>
  );
}
