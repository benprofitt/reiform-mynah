import { useState } from "react";
import HomePage from './pages/home_page'
import ProjectPage from './pages/project_page'


function App(): JSX.Element {
  const [isProjectOpen, setIsProjectOpen] = useState(false);

  return (
    <>
      {!isProjectOpen ? (
        <HomePage setIsProjectOpen={setIsProjectOpen} />
      ) : (
        <ProjectPage setIsProjectOpen={setIsProjectOpen} />
      )}
    </>
  );
}

export default App;
