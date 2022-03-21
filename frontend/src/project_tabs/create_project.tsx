import React from "react";
export default function CreateProject(): JSX.Element {
  return (
    <div className="grid grid-cols-2 divide-x divide-black h-full">
      <div>
        <div className="ml-4 font-bold text-lg">Project Settings</div>
        <div className="mb-3 pt-0">
          <input
            type="text"
            placeholder="Name"
            className="ml-4 mt-4 px-3 py-3 placeholder-black text-black rounded text-sm border-2 border-black outline-none focus:outline-none focus:ring w-2/3"
          />
        </div>
      </div>
      <div>
        <div className="ml-4 font-bold text-lg">Project Data</div>
        <div className="mt-4 ml-28 w-3/4 h-56 border-2 border-black place-content-center">
          <div className="h-8 text-center border-b-2 border-black">
            {" "}
            New Dataset{" "}
          </div>
          <div className="mt-4 ml-20 w-3/4 border-2 border-black text-center">
            {" "}
            Source{" "}
          </div>
          <div className="flex space-x-4">
            <input
              type="text"
              placeholder="Device"
              className="ml-24 mt-8 h-8 px-3 py-3 placeholder-black text-black rounded text-sm border-2 border-black outline-none focus:outline-none focus:ring w-1/2"
            />
            <button className="bg-blue-500 hover:bg-blue-700 text-white text-sm rounded px-2 h-8 mt-8">
              {" "}
              Choose{" "}
            </button>
          </div>
          <div className="flex space-x-4">
            <input
              type="text"
              placeholder="AWS: S3"
              className="ml-24 mt-4 h-8 px-3 py-3 placeholder-black text-black rounded text-sm border-2 border-black outline-none focus:outline-none focus:ring w-1/2"
            />
            <button className="bg-blue-500 hover:bg-blue-700 text-white text-sm rounded px-2 h-8 mt-4">
              {" "}
              Choose{" "}
            </button>
          </div>
        </div>
        <div className="mt-8 ml-28 w-3/4 h-96 border-2 border-black">
          <div className="h-8 text-center border-b-2 border-black">
            {" "}
            Choose Dataset{" "}
          </div>
          <div className="h-10 border-b-2 border-black flex flex-row space-x-64">
            <div className="ml-24 mt-2 h-6 w-16 text-center border-2 border-black">
              {" "}
              Name{" "}
            </div>
            <div className="text-center mt-2 w-32 h-6 border-2 border-black">
              Date Created
            </div>
          </div>
          <form>
            <div className="h-12 border-b-2 border-black">
              <div className="h-12 w-12 border-r-2 border-black">
                <div className="form-check">
                  <input
                    className="ml-4 mt-4 form-check-input appearance-none rounded-full h-4 w-4 border border-black checked:bg-blue-600 checked:border-blue-600 focus:outline-none transition duration-200 align-top bg-no-repeat bg-center bg-contain float-left mr-2 cursor-pointer"
                    type="radio"
                    name='source'
                    id="flexRadioDefault1"
                  />
                  <label className="form-check-label inline-block"></label>
                </div>
              </div>
            </div>
            <div className="h-12 border-b-2 border-black">
              <div className="h-12 w-12 border-r-2 border-black">
                <div className="form-check">
                  <input
                    className="ml-4 mt-4 form-check-input appearance-none rounded-full h-4 w-4 border border-black checked:bg-blue-600 checked:border-blue-600 focus:outline-none transition duration-200 align-top bg-no-repeat bg-center bg-contain float-left mr-2 cursor-pointer"
                    type="radio"
                    name="source"
                    id="flexRadioDefault2"
                  />
                  <label className="form-check-label inline-block"></label>
                </div>
              </div>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
