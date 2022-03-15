import HomePage from "./pages/home_page";
import ProjectPage from "./pages/project_page";
import LoginPage, { authCookieName } from "./pages/login_page";
import AccountSettingsPage from "./pages/account_settings_page";
import PageNotFound from "./pages/page_not_found";

// import {
//   BrowserRouter as Router,
//   Routes,
//   Route,
//   Navigate,
// } from "react-router-dom";
import Cookies from "universal-cookie";
import { Switch, Route, Redirect, Router } from "wouter";

function App(): JSX.Element {
  const cookies = new Cookies();
  const jwt: string = cookies.get(authCookieName);

  // maybe i should have a useeffect for login like halite instead of this boolean 

  return (
    <Router base="/mynah">
      {!Boolean(jwt) ? (
        <Switch>
          <Route path="/login" component={LoginPage} />
          <Route>
            <Redirect to="/login" />
          </Route>
        </Switch>
      ) : (
        <Switch>
          <Route path="/login">
            <Redirect to="/" />
          </Route>
          <Route path="/" component={HomePage} />
          <Route path="/project/:tab">
            {(params) => <ProjectPage route={params.tab} />}
          </Route>
          <Route path="/account-settings" component={AccountSettingsPage} />
          <Route component={PageNotFound} />
        </Switch>
      )}
    </Router>
  );
}

export default App;
