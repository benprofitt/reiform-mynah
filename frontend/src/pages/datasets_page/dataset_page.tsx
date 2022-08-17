import Ellipsis from "../../images/Ellipsis.svg";
import BackArrowIcon from "../../images/BackArrowIcon.svg";
import { Link, RouteComponentProps } from "wouter";
import { Tab } from "@headlessui/react";
import clsx from "clsx";
import { MynahICDataset } from "../../utils/types";
import { useQuery } from "react-query";
import makeRequest from "../../utils/apiFetch";
import { useState } from "react";
import Files from "./files";
import Reports from "./reports";
import ProcessDataModal from "./process_data_modal";

export default function DatasetPage(
  props: RouteComponentProps<{ uuid: string }>
): JSX.Element {
  const { data: datasets } = useQuery<MynahICDataset[]>("datasets", () =>
    makeRequest<MynahICDataset[]>("GET", "/api/v1/dataset/list")
  );
  const { uuid } = props.params;
  const [processDataOpen, setProcessDataOpen] = useState(false);
  const [pendingReports, setPendingReports] = useState<string[] | null>(null);
  if (datasets === undefined)
    return <div>Unable to retrive datasets, are you logged in?</div>;
  const dataset = datasets.find((dataset) => dataset.uuid === uuid);
  if (dataset === undefined)
    return (
      <div>
        This dataset either does not exist or you do not have permission to see
        it
      </div>
    );
  const { dataset_name: name } = dataset;
  return (
    <div className="flex h-screen flex-1">
      <Tab.Group as="div" className="w-full flex flex-col">
        <header className="w-full h-fit border-b border-grey1 pl-[32px] relative bg-white pt-[46px]">
          <h1 className="font-bold text-[28px]">{name}</h1>
          <Link to="/">
            <button className="flex items-center text-linkblue absolute left-[30px] top-[20px] font-bold">
              <img src={BackArrowIcon} alt="back arrow" className="mr-[2px]" />
              Back
            </button>
          </Link>
          <div className="absolute right-[36px] top-[60px] space-x-[20px] text-[18px] h-[38px] flex items-center">
            <button className="text-linkblue h-full w-[134px] text-center font-medium">
              Add More Data
            </button>
            <button
              className="bg-linkblue text-white h-full w-[134px] text-center font-medium rounded-md"
              onClick={() => setProcessDataOpen(true)}
            >
              Process Data
            </button>
            <ProcessDataModal
              isOpen={processDataOpen}
              close={() => setProcessDataOpen(false)}
              uuid={uuid}
              setPendingReports={setPendingReports}
            />
            <button className="h-full">
              <img src={Ellipsis} alt="more" />
            </button>
          </div>
          <Tab.List>
            {["Files", "Reports"].map((title) => (
              <Tab>
                {({ selected }) => (
                  <div
                    key={title}
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
        <main className="bg-grey w-full p-[32px] flex-1 overflow-y-scroll">
          <Tab.Panels>
            <Tab.Panel>
              <Files dataset={dataset} />
            </Tab.Panel>
            <Tab.Panel>
              <Reports dataset={dataset} pendingReports={pendingReports} />
            </Tab.Panel>
          </Tab.Panels>
        </main>
      </Tab.Group>
    </div>
  );
}
