import React from "react";
import Logo from "../images/ReiformLogo.png";

export interface ReiformLogoProps {
  className?: string;
  onClick?: () => void;
}

export default function ReiformLogo(props: ReiformLogoProps): JSX.Element {
  const {
    className = "",
    onClick = () => {
      // do nothing
    },
  } = props;
  return (
    <img
      className={className}
      onClick={onClick}
      src={Logo}
      alt="Reiform Logo"
    />
  );
}
