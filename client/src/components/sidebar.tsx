import { useState } from "react";
import { Link } from "react-router-dom";
import { ReactComponent as Logo } from "../assets/NewReiformLogo.svg";
import { ReactComponent as CreateNew } from "../assets/PlusCircle.svg";
import ImportData from "./import_data";

export default function Sidebar() {
  const [isAddingDataset, setIsAddingDataset] = useState(false)

  return (
    <>
    <nav className="h-screen bg-sideBar w-16 flex flex-col items-center">
      <Link to="/">
        <Logo className="w-[36px] h-[40px] mt-[18px]" />
      </Link>
      <button className="mt-[50px]" onClick={() => setIsAddingDataset(true)}>
        <CreateNew />
      </button>
    </nav>
    <ImportData open={isAddingDataset} close={() => setIsAddingDataset(false)}/>
    </>
    
  );
}
