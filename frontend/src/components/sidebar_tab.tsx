import clsx from "clsx";
import { Link } from "wouter";

export interface TabProps {
  title: string;
  path: string;
  selected?: boolean;
}

export default function SidebarTab(props: TabProps): JSX.Element {
  const { title, path, selected = false } = props;
  return (
    <Link to={path}>
      <div
        className={clsx(
          "rounded my-5 p-2 cursor-pointer w-24 h-16",
          selected ? "bg-green-300" : "border border-black"
        )}
      >
        {title}
      </div>
    </Link>
  );
}
