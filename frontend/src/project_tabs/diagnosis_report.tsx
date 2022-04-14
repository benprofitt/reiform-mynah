import { Menu } from "@headlessui/react";
import clsx from "clsx";
import React, { useState } from "react";
import CheckBoxes from "../components/checkbox";
import Dropdown from "../components/dropdown";

export default function DiagnosisReport(): JSX.Element {
  const all_options = ['Class 1', 'Class 2', 'Class 3', 'Class 4']
  const [selectedOptions, setSelectedOptions] = useState<string[]>([])
  const dropDownOptions = all_options.filter((option) => !selectedOptions.includes(option))

  return <div>
    <div className='mt-4 grid grid-cols-2 divide-x'>
      <div className='w-full h-40 border-2 border-black'>
        <div className='w-full h-8 border-b-2 border-black text-lg font-bold'> Metric Options </div>
        <div className='overflow-y-scroll'>
          <CheckBoxes Label='Only show bad images' />
          <CheckBoxes Label='Provide metrics as %-ages' />
        </div>
      </div>
      <div className='w-full h-40 border-2 border-black'>
        <div className='w-full border-b-2 border-black h-8 text-lg font-bold'>Filter By</div>
        <div className='flex flex-row gap-x-8'>
          <div className='mt-2 ml-4'>Add Class to Graph and Image Viewer:</div>
          <div className="flex justify-center">
            <div className="mt-2">
              <Menu>
              <Menu.Button className='bg-blue-500 hover:bg-blue-700 text-white text-sm px-4 py-2 rounded'>More</Menu.Button>
                <Menu.Items className='absolute w-full origin-top-right z-10'>
                {dropDownOptions.map((option, index) => (
                  <Menu.Item as = 'button' className='flex flex-row text-center bg-gray-200 hover:bg-gray-300 px-2' key = {index}
                  onClick = {() => {
                    console.log('clicked')
                    setSelectedOptions([...selectedOptions, option])
                  }}>
                    {option}
                  </Menu.Item>
                ))
                }
                </Menu.Items>
              </Menu>
            </div>
          </div>
        </div>
        <div className="mt-4 flex flex-row">
          {selectedOptions.map((option) => (
            <div className="flex flex-row" key = {option}>
              <div className={clsx('mt-4 ml-8 border-2 border-black h-6 w-16')}>
                  {option}
              </div>
              <button className="mt-4 bg-blue-500 hover:bg-blue-700 text-white text-sm px-2 h-6" onClick={() => {setSelectedOptions(selectedOptions.filter(item => item!== option))}}>
                -
              </button>
            </div>
          ))
          }
        </div>
      </div>
    </div>
  </div>
}