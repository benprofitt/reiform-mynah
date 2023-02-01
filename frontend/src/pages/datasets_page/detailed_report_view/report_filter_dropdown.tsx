import { Menu } from "@headlessui/react";
import clsx from "clsx";
import ArrowIcon from "../../../images/ArrowIcon.svg";
import MenuItem from "../../../components/menu_item";
import { MislabeledType } from "./image_list_viewer_and_scatter";

export default function ReportFilterDropdown({
  leftAligned,
  allClassNames,
  updateClassNamesFilter,
  updateMislabeledFilter,
  filteredClasses,
  mislabledFilterSetting,
  allowedMislabeledTypes
}: {
  leftAligned?: boolean;
  allClassNames: string[];
  updateClassNamesFilter: (className: string) => void;
  updateMislabeledFilter: (mislabledType: MislabeledType) => void;
  filteredClasses: string[];
  mislabledFilterSetting: MislabeledType[];
  allowedMislabeledTypes: MislabeledType[]
}): JSX.Element {
  console.log(filteredClasses)
  return (
    <Menu as="div" className="relative inline-block text-left z-I10">
      <div>
        <Menu.Button
          className={clsx(
            "inline-flex w-[114px] h-[40px] items-center rounded-md bg-white text-black focus:outline-none",
            !leftAligned && "px-[10px] border border-grey3"
          )}
        >
          Filter by
          <img src={ArrowIcon} alt="arrow" className="ml-auto mt-[4px]" />
        </Menu.Button>
      </div>
      <Menu.Items
        className={clsx(
          "absolute mt-2 w-56 divide-y divide-gray-100 rounded-md bg-white shadow-floating ring-1 ring-black ring-opacity-5 focus:outline-none z-30",
          leftAligned ? "origin-top-left left-0" : "origin-top-right right-0"
        )}
      >
        <Menu.Item as="div" className="flex flex-col bg-white z-30">
          Filter by class
          {allClassNames.map((className, ix) => (
            <label className="ml-[20px]" key={ix}>
              <input
                className="mr-[10px]"
                type="checkbox"
                checked={filteredClasses.includes(className)}
                onChange={() =>
                  updateClassNamesFilter(
                    className
                  )
                }
              />
              {className}
            </label>
          ))}
        </Menu.Item>
        <Menu.Item as="div" className="flex flex-col bg-white z-30">
        Filter by mislabled type
          {allowedMislabeledTypes.map((mislabeledType, ix) => (
            <label className="ml-[20px]" key={ix}>
              <input
                className="mr-[10px]"
                type="checkbox"
                checked={mislabledFilterSetting.includes(mislabeledType)}
                onChange={() =>
                  updateMislabeledFilter(mislabeledType)
                }
              />
              {mislabeledType}
            </label>
          ))}
        </Menu.Item>
      </Menu.Items>
    </Menu>
  );
}
