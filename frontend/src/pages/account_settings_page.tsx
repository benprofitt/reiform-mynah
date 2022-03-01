import React, { useState } from "react";
import Cookies from "universal-cookie";
import PageContainer from "../components/page_container";
import { authCookieName, authCookieOptions } from "./login_page";

export interface AccountSettingsPageProps {
  children?: JSX.Element;
}

export default function AccountSettingsPage(): JSX.Element {
  const cookies = new Cookies();
  const [isLoggingOut, setIsLoggingOut] = useState(false);
  const clearCookies = () => {
    cookies.remove(authCookieName, authCookieOptions);
    setIsLoggingOut(true);
    setTimeout(() => {
      window.location.reload();
    }, 500);
  };
  return (
    <PageContainer>
      <div className="mx-auto">
        <button
          className="my-10 w-48 h-10 border border-black"
          onClick={() => clearCookies()}
        >
          Clear JWT Cookie
        </button>
        {isLoggingOut ? <h3 className="text-center">Goodbye!</h3> : <></>}
      </div>
    </PageContainer>
  );
}
