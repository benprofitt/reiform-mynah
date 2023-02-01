import { Menu } from "@headlessui/react";
import clsx from "clsx";
import React, { ReactNode } from "react";
import MenuItem from "./menu_item";

export default function EllipsisMenu({children}: {children?: ReactNode}): JSX.Element {
    return <Menu
    as="div"
    className="absolute inline-block text-left right-[15px] top-[30%]"
  >
    {({ open }) => (
      <>
        <div>
          <Menu.Button
            className={clsx(
              "hover:bg-grey3 transition-colors duration-300 rounded-full w-[30px] aspect-square flex items-center justify-center group",
              open ? "bg-grey3" : "bg-clearGrey3"
            )}
          >
            {[0, 1, 2].map((ix) => (
              <div
                key={ix}
                className={clsx(
                  "rounded-full w-[4px] aspect-square mx-[2px] transition-colors duration-300 group-hover:bg-grey5 ",
                  open ? "bg-grey5" : "bg-grey6"
                )}
              />
            ))}
          </Menu.Button>
        </div>
        <Menu.Items className="z-10 absolute right-[15px] mt-[15px] w-56 origin-top-right rounded-md bg-white shadow-floating focus:outline-none">
          {children ?? <MenuItem text='View File Details' />}
        </Menu.Items>
      </>
    )}
  </Menu>
}