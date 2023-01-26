import DatasetsHomePage from "./pages/datasets_page/datasets_home_page";
import DatasetDetailPage from "./pages/datasets_page/dataset_detail_page/dataset_detail_page";
import LoginPage, { authCookieName } from "./pages/login_page";
import AccountSettingsPage from "./pages/account_settings_page/account_settings_page";

import Cookies from "universal-cookie";
import { Switch, Route, Redirect, Router } from "wouter";
import PageNotFound from "./pages/page_not_found";
import { QueryClient, QueryClientProvider, useQuery } from "react-query";
import SideBar from "./components/sidebar";
import ModelsHomePage from "./pages/models_page/models_home_page";
import ModelDetailPage from "./pages/models_page/models_detail_page/model_detail_page";

const queryClient = new QueryClient();

function App(): JSX.Element {
  const cookies = new Cookies();
  const jwt: string = cookies.get(authCookieName);
  // in actual production auth:
  // i will do the same thing, if no jwt just go to login page, but if there is a jwt, then before routing anywhere, we make a request to get datasets, and if the request returns bad jwt we gotta redirect
  console.log(jwt)
  return (
    <Router base="/mynah">
      {!Boolean(jwt) ? (
        <LoginPage />
      ) : (
        <QueryClientProvider client={queryClient}>
          <Switch>
            <Route path="/login">
              <Redirect to="/datasets" />
            </Route>
            <Route path="/">
              <Redirect to="/datasets" />
            </Route>
            <SideBar>
              <Switch>
                <Route path="/datasets" component={DatasetsHomePage} />
                <Route path="/datasets/ic/:uuid/:tab?/:id?/:type?" component={DatasetDetailPage} />
                {/* <Route path="/models" component={ModelsHomePage}/>
                <Route path="/models/:id/:tab?" component={ModelDetailPage}/> */}
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
