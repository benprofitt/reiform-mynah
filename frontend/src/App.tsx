import HomePage from './pages/home_page'
import ProjectPage from './pages/project_page'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'


function App(): JSX.Element {

  return (
    <Router>
      <Routes>
        <Route path='/mynah/project/*' element={<ProjectPage />} />
        <Route path='/mynah' element={<HomePage />} />
      </Routes>
    </Router>
  );
}

export default App;
