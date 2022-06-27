import Cookies from "universal-cookie";
import { authCookieName } from "../pages/login_page";
import axios, { Method } from "axios";

export default async function makeRequest<T>(
  method: Method,
  endpoint: string,
  body?: any,
  contentType?: string
) {
  const cookies = new Cookies();
  const jwt: string = cookies.get(authCookieName);
  return axios({
    // eslint-disable-next-line no-restricted-globals
    url: import.meta.env.DEV
      ? `http://localhost:8080${endpoint}`
      : `http://${location.host}${endpoint}`,
    method: method,
    headers: {
      "api-key": jwt,
      ...(contentType !== undefined ? { "Content-Type": contentType } : {}),
    },
    ...(body !== undefined ? { data: body } : {}),
  }).then((res) => {
    console.log(res);
    const data: T = res.data;
    return data;
  });
}