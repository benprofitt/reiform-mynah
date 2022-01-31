import { Switch } from "@headlessui/react";
import clsx from "clsx";
import React from "react";

function App() {
  const [isOn, setOn] = React.useState(false);
  const color = isOn ? "blue" : "red";
  return (
    <>
      <h1 className="text-center my-3">
        The humble beginnings of Reiform Mynah
      </h1>
      <hr className="border border-black mx-10" />
      <Switch
        className="flex gap-3 items-center justify-center mx-auto my-5"
        checked={isOn}
        onChange={() => setOn(!isOn)}
      >
        <p className="pointer-events-none">
          Click this text or the tile to change its color
        </p>
        <div
          className={clsx("w-10 h-10", isOn ? "bg-blue-400" : "bg-red-400")}
        />
      </Switch>
      <hr className="border border-black mx-48" />
      <p className="text-center mt-1">The tile is {color}</p>
    </>
  );
}

export default App;
