import { Tab } from "@headlessui/react";
import clsx from "clsx";
import { ReactNode } from "react";
import { useLocation } from "wouter";
import BackButton from "./back_button";

export interface DetailPageWithTabsProps {
  title: string;
  backButtonDestination: string;
  topRightButtons: ReactNode;
  tabNames: string[];
  tabPanels: ReactNode;
  selectedIndex: number;
  basePath: string;
}

export default function DetailPageWithTabs(props: DetailPageWithTabsProps) {
  const {
    title,
    backButtonDestination,
    topRightButtons,
    tabNames,
    tabPanels,
    selectedIndex,
    basePath,
  } = props;
  const [_location, setLocation] = useLocation();
  return (
    <div className="flex h-screen flex-1">
      <Tab.Group
        selectedIndex={selectedIndex}
        onChange={(index) => setLocation(`${basePath}/${tabNames[index]}`)}
      >
        <div className="w-full flex flex-col">
          <header className="w-full h-fit border-b border-grey1 pl-[32px] relative bg-white pt-[46px]">
            <BackButton
              className="absolute left-[30px] top-[20px]"
              destination={backButtonDestination}
            />
            <h1 className="font-bold text-[28px]">{title}</h1>
            <div className="absolute right-[36px] top-[60px] space-x-[20px] text-[18px] h-[38px] flex items-center">
              {topRightButtons}
            </div>
            <Tab.List>
              {tabNames.map((title) => (
                <Tab key={title} className="focus:outline-none">
                  {({ selected }) => (
                    <div
                      className={clsx(
                        "relative h-[40px] mr-[20px] mt-[20px] font-bold uppercase",
                        selected ? "text-black" : "text-grey2"
                      )}
                    >
                      {title}
                      {selected && (
                        <div className="absolute bottom-0 w-full bg-linkblue h-[5px] rounded-sm"></div>
                      )}
                    </div>
                  )}
                </Tab>
              ))}
            </Tab.List>
          </header>
          <main className="bg-grey w-full p-[32px] flex-1 overflow-y-clip">
            {tabPanels}
          </main>
        </div>
      </Tab.Group>
    </div>
  );
}
