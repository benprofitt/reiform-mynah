import clsx from "clsx";
import ReiformLogo from "./reiform_logo";
import { Link } from "wouter";
import { useState } from "react";
import ImportData from "./import_data";
import PlusCircle from "../images/PlusCircle.svg";
import HomeButton from "../images/HomeButton.svg";

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
  const { children } = props
  const [isAddingDataset, setIsAddingDataset] = useState(false);
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
        <SideBarButton onClick={() => setIsAddingDataset(true)}>
          <>
            <img src={PlusCircle} alt="add new dataset" />
            <ImportData
              open={isAddingDataset}
              close={() => setIsAddingDataset(false)}
            />
          </>
        </SideBarButton>
        <SideBarButton selected>
          <Link to="/home">
            <img src={HomeButton} alt="Go Home" />
          </Link>
        </SideBarButton>
        <Link
          to="/account-settings"
          className="w-[32px] aspect-square bg-green-700 grid place-items-center text-[12px] font-medium mt-auto rounded-full text-white"
        >
          JL
        </Link>
      </nav>
      {children}
    </div>
  );
}
