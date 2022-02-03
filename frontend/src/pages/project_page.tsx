import React, { useState } from "react";
import PageContainer from "../components/page_container";
import SideBar from "../components/sidebar";
import TopBar from "../components/topbar";

export interface ProjectPageProps {
  setIsProjectOpen: React.Dispatch<React.SetStateAction<boolean>>;
}

export default function ProjectPage(props: ProjectPageProps): JSX.Element {
  const { setIsProjectOpen } = props;
  const [projectTitle, setProjectTitle] = useState("Template Project Title");
  return (
    <PageContainer>
      <TopBar onClickLogo={() => setIsProjectOpen(false)}>
        <div className="border-r-2 border-black w-full p-2 pr-5 flex">
          <input
            className="font-bold ml-10 text-4xl p-2 border border-black w-full"
            value={projectTitle}
            onChange={(e) => setProjectTitle(e.target.value)}
          />
        </div>
        <div className="flex items-center justify-center p-5 w-fit mx-auto">
          <button className="border border-black w-36 h-full py-2">Next</button>
        </div>
      </TopBar>
      <div className="flex grow">
        <SideBar>
          <div className="p-2">Project Sidebar stuff</div>
        </SideBar>
        <p className="text-center mt-3 grow">Project Contents</p>
      </div>
    </PageContainer>
  );
}
