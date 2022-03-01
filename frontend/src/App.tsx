import HomePage from "./pages/home_page";
import ProjectPage from "./pages/project_page";
import LoginPage, { authCookieName } from "./pages/login_page";
import AccountSettingsPage from "./pages/account_settings_page";

import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import Cookies from "universal-cookie";

function App(): JSX.Element {
  const cookies = new Cookies();

  const loggedIn = cookies.get(authCookieName);

  console.log(loggedIn);

  return (
    <Router>
      <Routes>
        {!loggedIn ? (
          <>
            <Route path="/mynah/login" element={<LoginPage />} />
            <Route path="/*" element={<Navigate replace to="/mynah/login" />} />
          </>
        ) : (
          <>
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
            <Route path="/" element={<Navigate replace to={"/mynah"} />} />
          </>
        )}
      </Routes>
    </Router>
  );
}

export default App;
