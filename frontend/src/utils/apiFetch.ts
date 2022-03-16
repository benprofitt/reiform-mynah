import Cookies from "universal-cookie";
import { authCookieName } from "../pages/login_page";

export default async function makeRequest<T>(
  method: string,
  body: string,
  endpoint: string
) {
  const cookies = new Cookies();
  const jwt: string = cookies.get(authCookieName);
  const requestOptions: RequestInit = {
    method: method,
    headers: { "api-key": jwt, "Content-Type": "application/json" },
    body: body,
  };

  // eslint-disable-next-line no-restricted-globals
  return fetch(`http://${location.host}${endpoint}`, requestOptions)
    .then((res) => {
      console.log("got the res to json!", res);
      return res.json();
    })
    .then((res: T) => {
      console.log("valid response!", res);
      return res;
    });
}
