import { Menu } from "@headlessui/react";
import clsx from "clsx";

interface MenuItemProps {
    src?: string;
    text: string;
  }
  
export default function MenuItem(props: MenuItemProps): JSX.Element {
    const { src, text } = props;
    return (
      <Menu.Item>
        {({ active }) => (
          <button
            className={`${
              active ? "bg-sideBar text-white" : "text-gray-900"
            } flex w-full items-center rounded-md px-[20px] py-[15px]`}
          >
            {src && (
              <img
                src={src}
                alt={text}
                className={clsx(
                  "mr-[10px] h-[20px] aspect-square",
                  active && "invert"
                )}
              />
            )}
            {text}
          </button>
        )}
      </Menu.Item>
    );
  };