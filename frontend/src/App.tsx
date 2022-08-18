import DatasetsHomePage from "./pages/datasets_home_page";
import DatasetPage from "./pages/datasets_page/dataset_page";
import LoginPage, { authCookieName } from "./pages/login_page";
import AccountSettingsPage from "./pages/account_settings_page";

import Cookies from "universal-cookie";
import { Switch, Route, Redirect, Router } from "wouter";
import PageNotFound from "./pages/page_not_found";
import { QueryClient, QueryClientProvider, useQuery } from "react-query";
import SideBar from "./components/sidebar";

const queryClient = new QueryClient();

function App(): JSX.Element {
  const cookies = new Cookies();
  const jwt: string = cookies.get(authCookieName);
  console.log(jwt)
  return (
    <Router base="/mynah">
      {!Boolean(jwt) ? (
        <LoginPage />
      ) : (
        <QueryClientProvider client={queryClient}>
          <Switch>
            <Route path="/login">
              <Redirect to="/home" />
            </Route>
            <Route path="/">
              <Redirect to="/home" />
            </Route>
            <SideBar>
              <Switch>
                <Route path="/home" component={DatasetsHomePage} />
                <Route path="/dataset/ic/:uuid/:tab?/:id?/:type?" component={DatasetPage} />
                <Route
                  path="/account-settings"
                  component={AccountSettingsPage}
                />
                <Route component={PageNotFound} />
              </Switch>
            </SideBar>
          </Switch>
        </QueryClientProvider>
      )}
    </Router>
  );
}

export default App;
