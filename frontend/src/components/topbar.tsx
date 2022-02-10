import { Link } from "react-router-dom";
import ReiformLogo from "./reiform_logo";

export interface TopBarProps {
  children: JSX.Element | JSX.Element[];
}

export default function TopBar(props: TopBarProps): JSX.Element {
  const { children } = props;

  return (
    <div className="w-full h-20 border-b-2 border-black flex">
      <Link to="/mynah">
        <ReiformLogo className="ml-3 h-20 w-20" />
      </Link>
      {children}
    </div>
  );
}
