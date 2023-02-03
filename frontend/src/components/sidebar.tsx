import clsx from "clsx";
import ReiformLogo from "./reiform_logo";
import { Link, useLocation } from "wouter";
import { useState } from "react";
import ImportData from "./import_data";
import PlusCircle from "../images/PlusCircle.svg";
import DatasetsIcon from "../images/DatasetsIcon.svg";
import ModelsIcon from "../images/ModelsIcon.svg";
import CreateNewModel from "../pages/models_page/new_model_dialog";

export interface SideBarButtonProps {
  children?: JSX.Element;
  className?: string;
  selected?: boolean;
  onClick?: () => void;
}
export const SideBarButton = (props: SideBarButtonProps): JSX.Element => {
  const {
    children = <></>,
    className = "",
    selected = false,
    onClick = () => {},
  } = props;
  return (
    <button
      onClick={onClick}
      className={clsx(
        className,
        "grid place-items-center w-full h-[28px] hover:animate-pulse relative mb-[36px]"
      )}
    >
      {children}
      {selected && (
        <div className="absolute top-0 left-0 h-full w-[6px] bg-sidebarSelected" />
      )}
    </button>
  );
};

export interface SideBarProps {
  children: JSX.Element | JSX.Element[];
}
export default function SideBar(props: SideBarProps): JSX.Element {
  const { children } = props;
  const [loc, setLoc] = useLocation();
  const isDatasets = loc.startsWith("/datasets");
  const isModels = loc.startsWith("/models");
  console.log(loc);
  const [isAddingDataset, setIsAddingDataset] = useState(false);
  const [isCreatingModel, setIsCreatingModel] = useState(false);
  return (
    <div className="h-screen flex">
      <nav
        className={clsx(
          "w-[64px] h-screen flex flex-col items-center bg-sideBar py-[16px] shrink-0"
        )}
      >
        <Link to="/">
          <ReiformLogo className="w-[36px] mb-[60px]" />
        </Link>
        <SideBarButton
          onClick={() => {
            if (isDatasets) setIsAddingDataset(true);
            if (isModels) setIsCreatingModel(true);
          }}
        >
          <img src={PlusCircle} alt="add new dataset" />
        </SideBarButton>
        <Link to="/datasets">
          <SideBarButton selected={isDatasets}>
            <img src={DatasetsIcon} alt="Datasets" />
          </SideBarButton>
        </Link>
        {/* <Link to="/models">
          <SideBarButton selected={isModels}>
            <img src={ModelsIcon} alt="Datasets" />
          </SideBarButton>
        </Link> */}
        <Link
          to="/account-settings"
          className="w-[32px] aspect-square bg-green-700 grid place-items-center text-[12px] font-medium mt-auto rounded-full text-white"
        >
          RM
        </Link>
        {/* maybe just one stateful 'is creating' and then do the rest based on route */}
        <ImportData
          open={isAddingDataset}
          close={() => setIsAddingDataset(false)}
        />
        <CreateNewModel
          open={isCreatingModel}
          close={() => setIsCreatingModel(false)}
        />
      </nav>
      {children}
    </div>
  );
}
