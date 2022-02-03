import clsx from "clsx";
import ReiformLogo from "./reiform_logo";

export interface TopBarProps {
  children: JSX.Element | JSX.Element[];
  onClickLogo?: () => void;
}

export default function TopBar(props: TopBarProps): JSX.Element {
  const {
    children,
    onClickLogo = undefined,
  } = props;
  
  return (
    <div className="w-full h-20 border-b-2 border-black flex">
      <ReiformLogo className={clsx("ml-5", onClickLogo !== undefined ? 'cursor-pointer' : '')} onClick={onClickLogo} />
      {children}
    </div>
  );
}
