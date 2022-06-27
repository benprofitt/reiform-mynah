import clsx from "clsx";
import React, { useState } from "react";
import TopBar from "../components/topbar";
import SideBar from "../../components/sidebar";
import { Link } from "wouter";

type Options = "Projects" | "Datasets";

export default function HomePage(): JSX.Element {
  const [mode, setMode] = useState<Options>("Projects");
  const options: Options[] = ["Projects", "Datasets"];
  const headers = ["Name", "Datasets", "Date Created", "Download"];

  return (
    <>
      <TopBar>
        <div className="w-full flex h-full items-center">
          {options.map((options) => {
            return (
              <div
                className={clsx(
                  "h-14 w-2/5 border border-black ml-10 font-bold pl-4 flex items-center",
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
            to={mode === "Projects" ? "/project/create-project" : "/"}
            className="border border-black w-36 h-full py-2 text-center"
          >
            + New {mode.slice(0, -1)}
          </Link>
        </div>
      </TopBar>
      <div className="flex relative">
        <SideBar>
          {[0, 1, 2, 3, 4, 5].map((ix) => (
            <div
              key={ix}
              className={clsx(
                "w-full h-20 border-b-2 border-black flex items-center justify-center",
                ix === 0 ? "h-16" : "h-24" // first row shorter than the rest
              )}
            >
              {ix !== 0 && (
                <div className="w-6 h-6 shrink-0 border-2 border-black" />
              )}
            </div>
          ))}
        </SideBar>
        {/* if i wanna make this x scrollable i need breakpoints, where i'll have it be scrollable when skinny but once its wide enough to fit it,
        it'll witch from w-fit to w-full*/}
        <div className="w-full">
          {[0, 1, 2, 3, 4, 5].map((ix) => (
            <div
              key={ix}
              className={clsx(
                "w-full h-20 border-b-2 border-black flex items-center",
                ix === 0 ? "h-16" : "h-24" // first row shorter than the rest
              )}
            >
              {ix === 0 ? ( // only show header names on the first row
                <div className="ml-24 mr-10 flex justify-between w-full /space-x-20">
                  {headers.map((header) => (
                    <div className="w-40" key={header}>
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
              ) : (
                <div className="ml-24 mr-10 flex justify-between w-full">
                  {headers.map((header) => (
                    <div className="w-40" key={header}>
                      <div
                        className={clsx(
                          "text-center",
                          mode === "Datasets" &&
                            ["Datasets", "Download"].includes(header) &&
                            "hidden"
                        )}
                      >
                        ...
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
