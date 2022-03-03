import react from "react";
import { Link } from "react-router-dom";
import ReiformLogo from "../components/reiform_logo";

export default function PageNotFound(): JSX.Element {
  return (
    <div id="wrapper">
      <div className="mt-[200px] flex space-x-4 justify-center items-center">
        <Link to="/mynah">
          <ReiformLogo className="h-20 w-20" />
        </Link>
        <p className="text-6xl text-center"> 404 </p>
      </div>
      <p className="text-4xl text-center">This page could not be found</p>
    </div>
  );
}
