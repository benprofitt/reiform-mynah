import { Menu } from "@headlessui/react";
import clsx from "clsx";
import React, { useState } from "react";
import CheckBoxes from "../components/checkbox";
import Dropdown from "../components/dropdown";
import Table from "../components/table";
import Histogram from "../components/histogram";
import Scatterplot from "../components/scatterplot";
import ImageGallery from "react-image-gallery";
import "react-image-gallery/styles/css/image-gallery.css";

export default function DiagnosisReport(): JSX.Element {
  const all_options = ["Class 1", "Class 2", "Class 3", "Class 4"];
  const [selectedOptions, setSelectedOptions] = useState<string[]>([]);
  const dropDownOptions = all_options.filter(
    (option) => !selectedOptions.includes(option)
  );

  const color = ["red", "blue", "orange"];
  const images = [
    {
      original: "https://picsum.photos/id/1018/1000/600/",
      thumbnail: "https://picsum.photos/id/1018/250/150/",
    },
    {
      original: "https://picsum.photos/id/1015/1000/600/",
      thumbnail: "https://picsum.photos/id/1015/250/150/",
    },
    {
      original: "https://picsum.photos/id/1019/1000/600/",
      thumbnail: "https://picsum.photos/id/1019/250/150/",
    },
  ];

  const mydata = [
    {
      class: "class1",
      points: [
        { x: 100, y: 200 },
        { x: 120, y: 100 },
        { x: 170, y: 300 },
        { x: 140, y: 250 },
        { x: 150, y: 400 },
        { x: 110, y: 280 },
      ],
      bad: 129,
      acceptable: 905,
    },
    {
      class: "class2",
      bad: 20,
      points: [
        { x: 300, y: 300 },
        { x: 400, y: 500 },
        { x: 200, y: 700 },
        { x: 340, y: 350 },
        { x: 560, y: 500 },
        { x: 230, y: 780 },
        { x: 500, y: 400 },
        { x: 300, y: 500 },
        { x: 240, y: 300 },
        { x: 320, y: 550 },
        { x: 500, y: 400 },
        { x: 420, y: 280 },
      ],
      acceptable: 50,
    },
    {
      class: "class3",
      points: [
        { x: 124, y: 200 },
        { x: 198, y: 100 },
        { x: 109, y: 300 },
        { x: 99, y: 250 },
        { x: 150, y: 400 },
        { x: 102, y: 280 },
        { x: 340, y: 313 },
        { x: 560, y: 542 },
        { x: 230, y: 724 },
        { x: 500, y: 420 },
      ],
      bad: 120,
      acceptable: 90,
    },
  ];

  return (
    <div>
      <div className="mt-4 grid grid-cols-2 divide-x">
        <div className="w-full h-40 border-2 border-black">
          <div className="w-full h-8 border-b-2 border-black text-lg font-bold">
            {" "}
            Metric Options{" "}
          </div>
          <div className="overflow-y-scroll">
            <CheckBoxes Label="Only show bad images" />
            <CheckBoxes Label="Provide metrics as %-ages" />
          </div>
        </div>
        <div className="w-full h-40 border-2 border-black">
          <div className="w-full border-b-2 border-black h-8 text-lg font-bold">
            Filter By
          </div>
          <div className="flex flex-row gap-x-8">
            <div className="mt-2 ml-4">
              Add Class to Graph and Image Viewer:
            </div>
            <div className="flex justify-center">
              <div className="mt-2">
                <Menu>
                  <Menu.Button className="bg-blue-500 hover:bg-blue-700 text-white text-sm px-4 py-2 rounded">
                    More
                  </Menu.Button>
                  <Menu.Items className="absolute w-full origin-top-right z-10">
                    {dropDownOptions.map((option, index) => (
                      <Menu.Item
                        as="button"
                        className="flex flex-row text-center bg-gray-200 hover:bg-gray-300 px-2"
                        key={index}
                        onClick={() => {
                          console.log("clicked");
                          setSelectedOptions([...selectedOptions, option]);
                        }}
                      >
                        {option}
                      </Menu.Item>
                    ))}
                  </Menu.Items>
                </Menu>
              </div>
            </div>
          </div>
          <div className="mt-4 flex flex-row">
            {selectedOptions.map((option) => (
              <div className="flex flex-row" key={option}>
                <div
                  className={clsx("mt-4 ml-8 border-2 border-black h-6 w-16")}
                >
                  {option}
                </div>
                <button
                  className="mt-4 bg-blue-500 hover:bg-blue-700 text-white text-sm px-2 h-6"
                  onClick={() => {
                    setSelectedOptions(
                      selectedOptions.filter((item) => item !== option)
                    );
                  }}
                >
                  -
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div>
        <div className="border-b-2 border-black"></div>
        <div className="grid grid-cols-2 divide-black divide-x-2 border-2 border-black items-center justify-center w-[85vw] mx-auto mt-2 h-[59vh] text-lg indent-2">
          <div>
            {" "}
            Image Counts Per Class
            <div className="border-2 border-black ml-1 w-[40vw] mx-auto mt-2 h-[53vh]">
              All Matrics are given here:
              <Table
                header={["Breakdown", " ", " ", " "]}
                data={mydata}
                keys={["Class", "Bad", "Acceptable", "Total"]}
              />
              <div className="text-base mt-">Visual Breakdown:</div>
              <Histogram Xaxis="class" data={mydata} />
            </div>
          </div>
          <div className="h-full text-xl">
            Outlier Visualization
            <Scatterplot Xaxis="x" Yaxis="y" data={mydata} colors={color} />
            <div className="border border-black border-b-1" />
            <ImageGallery
              items={images}
              showPlayButton={false}
              showFullscreenButton={false}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
