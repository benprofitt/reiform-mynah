import { Link } from "wouter";
import ReiformLogo from "./reiform_logo";

export interface TopBarProps {
  children: JSX.Element | JSX.Element[];
}

export default function TopBar(props: TopBarProps): JSX.Element {
  const { children } = props;

  return (
    <div className="w-full h-20 flex pr-3 mt-3">
      <Link to="/">
        <ReiformLogo className="ml-6 h-20 w-20 mr-6" />
      </Link>

      <div className="border border-black w-full ml-4 flex items-center justify-between px-10 rounded">
        {children}
      </div>
    </div>
  );
}
