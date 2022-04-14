import React from 'react';
import { Menu } from "@headlessui/react";
import Dropdown from '../components/dropdown';

export default function DataCleaning(): JSX.Element {
  const menuOptions = ['1', '2', '3', '4', '5']
  return (
    <div>
      <div className='mt-8 font-lg'>Choose the issues you would like us to correct based on the diagnoses:</div>
      <div className='mt-8 border-2 border-black w-full h-2'></div>
      <form>
        <div className='fex flex-row'>
          <div className="form-check">
            <input
              className="ml-4 mt-8 form-check-input appearance-none rounded-full h-4 w-4 border border-black checked:bg-blue-600 checked:border-blue-600 focus:outline-none transition duration-200 align-top bg-no-repeat bg-center bg-contain float-left mr-2 cursor-pointer"
              type="radio"
              name='source'
              id="flexRadioDefault1"
            />
            <label className="form-check-label inline-block"></label>
          </div>
          <div>Auto-clean (Recommended)</div>
        </div>
        <div className='mt-8 w-full h-2 border-2 border-black ml-4'></div>
        <div className='flex flex-row'>
          <div className="form-check">
            <input
              className="ml-4 mt-8 form-check-input appearance-none rounded-full h-4 w-4 border border-black checked:bg-blue-600 checked:border-blue-600 focus:outline-none transition duration-200 align-top bg-no-repeat bg-center bg-contain float-left mr-2 cursor-pointer"
              type="radio"
              name="source"
              id="flexRadioDefault2"
            />
            <label className="form-check-label inline-block"></label>
          </div>
          <div className='mt-6'>Manual selection</div>
        </div>
      </form>
      <div className='mt-8 w-full h-8 border-2 border-black flex justify-between'>
        <div>Filter Data to Clean</div>
        <button className="bg-blue-500 hover:bg-blue-700 text-white text-sm rounded px-2 h-6 mt-0.5 mr-4">
          +Add Cleaning Row
        </button>
      </div>
      <div className='border-2 border-black w-full h-96 flex flex-row'>
        <div className='border-b-2 border-black w-11/12 h-20'>
          <div className='mt-4 flex space-x-8'>
            <div>Issue Category</div>
            <Dropdown text={'Stats'} Labels={menuOptions} />
          </div>
          <div className='flex space-x-24 mt-2'>
            <div>Issue</div>
            <Dropdown text={'Labelling'} Labels={menuOptions} />
          </div>
        </div>
        <div className='border-x-2 border-black w-1/12 h-full'></div>
      </div>
    </div>
  );
}