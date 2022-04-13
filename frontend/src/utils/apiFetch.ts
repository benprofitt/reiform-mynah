import Cookies from "universal-cookie";
import { authCookieName } from "../pages/login_page";
import axios, { Method } from "axios";

export default async function makeRequest<T>(
  method: Method,
  body: any,
  endpoint: string,
  contentType: string,
) {
  const cookies = new Cookies();
  const jwt: string = cookies.get(authCookieName);
  return axios(({
    // eslint-disable-next-line no-restricted-globals
    url: `http://${location.host}${endpoint}`,
    method: method,
    headers: { "api-key": jwt, "Content-Type": contentType },
    data: body
  }))
    .then((res) => {
      const data: T = res.data
      return data
    })
}
