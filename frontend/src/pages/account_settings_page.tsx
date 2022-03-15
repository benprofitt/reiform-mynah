import React, { useState } from "react";
import Cookies from "universal-cookie";
import PageContainer from "../components/page_container";
import { authCookieName, authCookieOptions } from "./login_page";
import AddUserModal from "../components/add_user_modal";

export interface AccountSettingsPageProps {
  children?: JSX.Element;
}

export default function AccountSettingsPage(): JSX.Element {
  const cookies = new Cookies();
  const [isLoggingOut, setIsLoggingOut] = useState(false);
  const clearCookies = () => {
    cookies.remove(authCookieName, authCookieOptions);
    setIsLoggingOut(true);
    setTimeout(() => window.location.reload(), 300);
  };

  const [isModalOpen, setModalOpen] = useState(false);

  return (
    <PageContainer>
      <div className="mx-auto flex flex-col">
        <button
          className="my-10 w-48 h-10 border border-black"
          onClick={() => clearCookies()}
        >
          Clear JWT Cookie
        </button>
        <button
          className="my-10 w-48 h-10 border border-black"
          onClick={() => setModalOpen(true)}
        >
          + Add User
        </button>
        {isLoggingOut ? <h3 className="text-center">Goodbye!</h3> : <></>}
      </div>
      <AddUserModal open={isModalOpen} onClose={() => setModalOpen(false)} />
    </PageContainer>
  );
}
