import clsx from "clsx";
import React, { useState } from "react";
import PageContainer from "./../components/page_container";
import TopBar from "../components/topbar";
import SideBar from "../components/sidebar";
import { Link } from "wouter";

type Options = "Projects" | "Datasets";

export default function HomePage(): JSX.Element {
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
          {/* this will change in the future to accomodate a model for datasets,
          and will also eventually be generating unique project ids */}
          <Link
            to={mode === "Projects" ? "/project/data-ingest" : "/"}
            className="border border-black w-36 h-full py-2 text-center"
          >
            + New {mode.slice(0, -1)}
          </Link>
        </div>
      </TopBar>
      <div className="flex grow relative">
        {/* All the sidebar is doing here is making the vertical line.
        Since the elemenets in the main section stretch over and across,
        the bar is just here for show. */}
        <SideBar />
        <div className="w-full absolute">
          {[0, 1, 2, 3, 4, 5].map((ix) => (
            <div
              key={ix}
              className={clsx(
                "w-full h-20 border-b-2 border-black flex items-center",
                ix === 0 ? "h-16" : "h-24" // first row shorter than the rest
              )}
            >
              <div className="ml-[35px] w-6 h-6 shrink-0 border-2 border-black" />
              {ix === 0 && ( // only show header names on the first row
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
