import { Menu } from "@headlessui/react";
import clsx from "clsx";
import ArrowIcon from "../images/ArrowIcon.svg"
import MenuItem from "./menu_item";

export default function FilterDropdown({leftAligned}: {leftAligned?: boolean;}): JSX.Element {
    return (
      <Menu as="div" className="relative inline-block text-left z-10">
        <div>
          <Menu.Button className={clsx("inline-flex w-[114px] h-[40px] items-center rounded-md bg-white text-black focus:outline-none", !leftAligned && 'px-[10px] border border-grey3')}>
            Filter by
            <img src={ArrowIcon} alt="arrow" className="ml-auto mt-[4px]" />
          </Menu.Button>
        </div>
        <Menu.Items className={clsx("absolute mt-2 w-56 divide-y divide-gray-100 rounded-md bg-white shadow-floating ring-1 ring-black ring-opacity-5 focus:outline-none", leftAligned ? 'origin-top-left left-0' : 'origin-top-right right-0')}>
          <MenuItem text="Date" />
          <MenuItem text="Size" />
        </Menu.Items>
      </Menu>
    );
  };