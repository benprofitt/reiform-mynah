import axios, { Method } from "axios";

export default async function makeRequest<T>(
  method: Method,
  endpoint: string,
  body?: any,
  contentType?: string
) {
  return axios<T>({
    // eslint-disable-next-line no-restricted-globals
    url: import.meta.env.DEV
      ? `http://localhost:8080${endpoint}`
      : `http://${location.host}${endpoint}`,
    method: method,
    headers: {
      ...(contentType !== undefined ? { "Content-Type": contentType } : {}),
    },
    ...(body !== undefined ? { data: body } : {}),
  }).then((res) => {
    console.log(res);
    return res.data;
  });
}