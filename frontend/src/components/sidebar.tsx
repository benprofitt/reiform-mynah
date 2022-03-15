import React from "react";
import { Link } from "wouter";

export interface SideBarProps {
  children?: JSX.Element | JSX.Element[];
}

export default function SideBar(props: SideBarProps): JSX.Element {
  const { children = <></> } = props;
  return (
    <div className="w-24 h-full shrink-0 object-contain border-r-2 border-black flex flex-col relative text-center pt-4">
      {children}
      <Link to="/account-settings">
        <div className="absolute bottom-0 h-fit border-t-2 border-black p-2">
          Account Settings
        </div>
      </Link>
    </div>
  );
}
