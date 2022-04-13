import React, { useState } from "react";
import makeRequest from "../utils/apiFetch";
import {
  CreateICProject,
  MynahICDataset,
  MynahICProject,
} from "../utils/types";
import ImportData from "../components/import_data";
import ChooseDataset from "../components/choose_dataset";
import NextButton from "../components/next_button";
import { useLocation } from "wouter";
declare module "react" {
  interface InputHTMLAttributes<T> extends HTMLAttributes<T> {
    // extends React's HTMLAttributes
    directory?: string;
    webkitdirectory?: string;
  }
}

export interface CreateProjectProps {
  nextPath: string;
}

export default function CreateProject(props: CreateProjectProps): JSX.Element {
  const { nextPath } = props;

  const [, setLocation] = useLocation();

  // this will come from the backend in the future
  const [datasets, setDatasets] = useState<MynahICDataset[]>([]);

  // project creation state
  const [projectName, setProjectName] = useState("");
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);

  // project submission
  const toSubmit: CreateICProject = {
    name: projectName,
    datasets: selectedDatasets,
  };
  const isValid = toSubmit.name.length > 3 && toSubmit.datasets.length > 0;

  return (
    <div className="grid grid-cols-2 divice-x-r divide-black h-full">
      <NextButton
        onClick={() => {
          makeRequest<MynahICProject>(
            "POST",
            toSubmit,
            "/api/v1/icproject/create",
            "application/json"
          )
            .then((res) => {
              setLocation(nextPath);
            })
            .catch((err) => {
              alert('err creating project')
              console.log(err);
            });
        }}
        text="Create Project"
        active={isValid}
      />
      {/* left side */}
      <div className="flex flex-col items-start w-full px-4">
        <h1 className="font-bold text-lg">Project Settings</h1>
        <input
          value={projectName}
          onChange={(e) => setProjectName(e.target.value)}
          type="text"
          placeholder="Name"
          className="mt-4 px-3 py-3 placeholder-black text-black rounded text-sm border-2 border-black outline-none focus:outline-none focus:ring w-96"
        />
      </div>

      {/* right side */}
      <div className="flex flex-col justify-center items-center w-full px-4 border-black border-l">
        <h1 className="font-bold text-lg w-full text-left">Project Data</h1>
        <ImportData
          setDatasets={setDatasets}
          setSelectedDatasets={setSelectedDatasets}
        />
        <ChooseDataset
          selectedDatasets={selectedDatasets}
          setSelectedDatasets={setSelectedDatasets}
          datasets={datasets}
        />
      </div>
    </div>
  );
}
