import React from "react";
export default function DataEgress(): JSX.Element {
  return (
    <div className="flex mx-auto items-center justify-center space-x-10 w-full h-full">
      <div className="border-2 border-black w-96">
        <div className="h-16 border-b-2 border-black text-center text-lg">
          Export Data
        </div>
        <div className="mt-10 mx-auto w-5/6 h-10 border-2 border-black text-center">
          Save to:
        </div>
        <div className="flex space-x-4 mx-auto mt-4 w-fit">
          <div className=" w-40 h-8 border-2 border-black">
            Device
          </div>
          <button className="bg-blue-500 hover:bg-blue-700 text-white text-sm px-2 rounded">
            Choose
          </button>
        </div>
        <div className="flex space-x-4 mx-auto my-4 w-fit">
          <div className="w-40 h-8 border-2 border-black">
            AWS: S3
          </div>
          <button className="bg-blue-500 hover:bg-blue-700 text-white text-sm px-2 rounded">
            Choose
          </button>
        </div>
      </div>
      ​
      <div className="border-2 border-black divide-y-2 divide-black w-96">
        <div className="text-center h-16 text-lg">Choose Dataset</div>
        <div className="h-10">
          <div className="grid grid-cols-2 ml-20 h-10 place-content-center">
            <div className="border border-black w-2/4 h-7 text-center">
              Name
            </div>
            <div className="border border-black w-3/4 h-7 text-center">
              Data Created
            </div>
          </div>
          <div className="mt-0 h-32 w-16 border-r-2 border-black" />
        </div>
        <div className="h-16">
          <div className="ml-4 mt-4 w-7 h-7 border-2 border-black" />
        </div>
        <div className="h-16">
          <div className="ml-4 mt-4 w-7 h-7 border-2 border-black" />
        </div>
        <div className="h-80"></div>
      </div>
    </div>
  );
}
