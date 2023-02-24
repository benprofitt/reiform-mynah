import { Dialog } from "@headlessui/react";
import clsx from "clsx";
import { Dispatch, SetStateAction, useState } from "react";
import { useQuery } from "react-query";
import makeRequest from "../../../utils/apiFetch";
import { AsyncTaskData } from "../../../utils/types";

export interface ProcessDataModalProps {
  uuid: string;
  isOpen: boolean;
  close: () => void;
}

interface StartingJobResponse {
  task_uuid: string;
}

export default function ProcessDataModal(
  props: ProcessDataModalProps
): JSX.Element {
  const { uuid, isOpen, close } = props;
  const [selected, setSelected] = useState<"correction" | "diagnosis" | null>(
    null
  );
  const [labelingErrors, setLabelingErrors] = useState(false);
  const [intraclassVariance, setIntraclassVariance] = useState(false);

  const onClose = () => {
    setSelected(null);
    setLabelingErrors(false);
    setIntraclassVariance(false);
    close();
  };

  const isValid =
    selected === "diagnosis" ||
    (selected && (labelingErrors || intraclassVariance));

  const { refetch } = useQuery<AsyncTaskData[]>("tasks", () =>
    makeRequest<AsyncTaskData[]>("GET", "/api/v1/task/list")
  );
  
  const submitRequest = () => {
    if (!isValid) return;
    const types: string[] = [];
    switch (selected) {
      case "diagnosis": {
        types.push(
          "ic::diagnose::mislabeled_images",
          "ic::diagnose::class_splitting"
        );
        break;
      }
      case "correction": {
        if (labelingErrors) types.push("ic::correct::mislabeled_images");
        if (intraclassVariance) types.push("ic::correct::class_splitting");
        break;
      }
      default: {
        throw new Error("invalid request");
      }
    }
    const body = {
      tasks: types,
      dataset_uuid: uuid,
    };
    makeRequest<StartingJobResponse>(
      "POST",
      "/api/v1/dataset/ic/process/start",
      body
    ).then((x) => {
      refetch();
    });
  };
  return (
    <Dialog
      open={isOpen}
      onClose={onClose}
      className="fixed inset-0 w-full h-full flex items-center justify-center"
    >
      <Dialog.Overlay className="w-full h-full bg-black absolute  top-0 left-0 opacity-20 z-0" />
      <main className="bg-white w-[528px] relative z-10 p-[30px]">
        <button
          className="absolute top-[15px] right-[20px] text-[40px] leading-none"
          onClick={onClose}
        >
          x
        </button>
        <Dialog.Title className="text-3xl">Process data</Dialog.Title>
        {<form
          className="flex flex-col text-[18px]"
          onSubmit={(e) => {
            e.preventDefault();
            submitRequest();
            onClose();
          }}
        >
          <h2 className="font-bold my-[10px]">Select process options</h2>
          <label>
            <input
              className="mr-[10px]"
              type="radio"
              value="diagnosis"
              checked={selected === "diagnosis"}
              onChange={() => setSelected("diagnosis")}
            />
            Diagnosis
          </label>
          <label>
            <input
              className="mr-[10px]"
              type="radio"
              value="correction"
              checked={selected === "correction"}
              onChange={(e) => {
                setSelected("correction");
                setLabelingErrors(true);
                setIntraclassVariance(true);
              }}
            />
            Correction
          </label>
          <label className="ml-[20px]">
            <input
              className="mr-[10px]"
              disabled={selected !== "correction"}
              type="checkbox"
              checked={selected === "correction" && labelingErrors}
              onChange={() => setLabelingErrors((cur) => !cur)}
            />
            Labeling errors
          </label>
          <label className="ml-[20px]">
            <input
              className="mr-[10px]"
              disabled={selected !== "correction"}
              type="checkbox"
              checked={selected === "correction" && intraclassVariance}
              onChange={() => setIntraclassVariance((cur) => !cur)}
            />
            Intra-class variance
          </label>
          <button
            type="submit"
            className={clsx(
              "w-full h-[40px] text-white mt-[30px] font-bold",
              !isValid ? "bg-grey1" : "bg-blue-600"
            )}
          >
            Start process
          </button>
        </form>}
      </main>
    </Dialog>
  );
}
