import HomePage from "./pages/home_page";
import ProjectPage from "./pages/project_page";
import LoginPage, { authCookieName } from "./pages/login_page";
import AccountSettingsPage from "./pages/account_settings_page";
import PageNotFound from './pages/page_not_found'

import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import Cookies from "universal-cookie";

function App(): JSX.Element {
  const cookies = new Cookies();
  const jwt: string = cookies.get(authCookieName);

  return (
    <Router>
      <Routes>
        {!Boolean(jwt) ? (
          <>
            <Route path="*" element={<Navigate replace to="/mynah/login" />} />
            <Route path="/mynah/login" element={<LoginPage />} />
          </>
        ) : (
          <>
            <Route path="*" element={<PageNotFound />} />
            <Route
              path="/mynah/login"
              element={<Navigate replace to="/mynah" />}
            />
            <Route path="/mynah" element={<HomePage />} />
            <Route path="/mynah/project/*" element={<ProjectPage />} />
            <Route
              path="/mynah/account-settings"
              element={<AccountSettingsPage />}
            />
          </>
        )}
      </Routes>
    </Router>
  );
}

export default App;
