import React from "react";
import { Menu } from "@headlessui/react";

export interface DropdownProps {
  Labels: string[];
  text: string;
}

export default function Dropdown(props: DropdownProps): JSX.Element {
  const { Labels, text } = props;
  return (
    <Menu as="div" className="relative inline-block text-left">
      <Menu.Button className='bg-blue-500 hover:bg-blue-700 text-white text-sm px-2 rounded'>
        {text}
      </Menu.Button>
      <Menu.Items className='absolute w-full origin-top-right z-10'>
        {Labels.map((option, index) => (
          <Menu.Item as='button' className='flex flex-row text-center bg-gray-200 w-full hover:bg-gray-300 px-2' key={index}
            onClick={() => {
              console.log('clicked')
            }}>
            {option}
          </Menu.Item>
        ))
        }
      </Menu.Items>
    </Menu>
  );
}