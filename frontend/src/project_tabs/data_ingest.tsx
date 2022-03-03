import React from "react";

export default function DataIngest(): JSX.Element {
  return (
    <div className="grid grid-cols-2 mt-20 ml-4 mr-4 space-x-4 place-self-center">
      <div className="border-2 border-black">
        <div className="w-90 h-16 border-b-2 border-black text-center text-lg">
          Import Data
        </div>
        <div className="mt-10 mx-auto w-80 h-10 border-2 border-black text-center">
          Source
        </div>
        <div className="mt-4 mx-auto w-64 h-8 border-2 border-black">
          Device
        </div>
        <div className="mt-4 mx-auto w-64 h-8 border-2 border-black">
          AWS: S3
        </div>
        <div className="mt-4 mx-auto w-64 h-8 border-2 border-black">Azure</div>
      </div>

      <div className="border-2 border-black divide-y-2 divide-black">
        <div className="text-center h-16 text-lg">Choose Dataset</div>
        <div className="h-10">
          <div className="grid grid-cols-2 ml-20 h-10 place-content-center">
            <div className="border border-black w-20 h-7 text-center">Name</div>
            <div className="border border-black w-40 h-7 text-center">
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
