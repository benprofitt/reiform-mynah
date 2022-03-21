import clsx from "clsx";
import React from "react";
import { Link } from "wouter";
import SidebarTab from "./sidebar_tab";

export interface SideBarProps {
  children?: JSX.Element | JSX.Element[];
  homePage?: boolean;
}

export default function SideBar(props: SideBarProps): JSX.Element {
  const { children = <></>, homePage = false } = props;
  return (
    <div
      className={clsx(
        "w-32 shrink-0 h-full flex flex-col items-center",
        homePage && "border-r-2 border-black"
      )}
    >
      {children}
      <SidebarTab path="/account-settings" title="Account Settings" />
    </div>
  );
}
