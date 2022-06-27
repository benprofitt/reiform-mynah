import { useQuery } from "react-query";
import Cookies from "universal-cookie";
import { authCookieName } from "../pages/login_page";

const cookies = new Cookies();
const jwt: string = cookies.get(authCookieName);
const headers = new Headers();
headers.set("api-key", jwt);

export interface ImageProps {
  src: string
  className: string
}

export default function Image(props: ImageProps): JSX.Element {
  const { src: endpoint, className } = props
  const {
    error,
    isLoading,
    data: src,
  } = useQuery(endpoint, () =>
    getImg(endpoint)
  );
  return <img alt="img" className={className} src={src} />
}

async function getImg(url: string) {
  const response = await fetch(
    import.meta.env.DEV
      ? `http://localhost:8080${url}`
      : `http://${location.host}${url}`,
    { headers }
  );
  const blob = await response.blob();
  const objectUrl = URL.createObjectURL(blob);
  return objectUrl;
}