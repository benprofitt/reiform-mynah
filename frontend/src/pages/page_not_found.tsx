import { Link } from "wouter";
import ReiformLogo from "../components/reiform_logo";

export default function PageNotFound(): JSX.Element {
  return (
    <div className="flex flex-col flex-1 h-screen justify-center items-center">
      <div className="flex flex-row w-full items-center justify-center">
        <Link to="/">
          <ReiformLogo className="aspect-square w-[70px]" />
        </Link>
        <h1 className="text-6xl text-center ml-[20px]">404</h1>
      </div>
      <h2 className="text-4xl text-center">This page could not be found</h2>
    </div>
  );
}
