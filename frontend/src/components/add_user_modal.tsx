import { Dialog } from "@headlessui/react";
import clsx from "clsx";
import React, { useEffect, useRef, useState } from "react";
import Cookies from "universal-cookie";
import { authCookieName } from "../pages/login_page";

export interface AddUserModalProps {
  open: boolean;
  onClose: () => void;
}

interface User {
  uuid: string;
  name_first: string;
  name_last: string;
}

interface CreateUserResponse {
  jwt: string;
  user: User;
}

export default function AddUserModal(props: AddUserModalProps): JSX.Element {
  const { open, onClose: onCloseProp } = props;
  const cancelButtonRef = useRef(null);
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [displayedJWT, setDisplayedJWT] = useState("");
  const [awaitingJWT, setAwaitingJWT] = useState(false);
  const cookies = new Cookies();
  const jwt: string = cookies.get(authCookieName);

  const isValid = Boolean(firstName) && Boolean(lastName);

  const onSubmit = (e: React.MouseEvent<HTMLButtonElement, MouseEvent>) => {
    e.preventDefault();
    if (!isValid || awaitingJWT) return;
    setAwaitingJWT(true);
    const requestOptions: RequestInit = {
      method: "POST",
      headers: { "api-key": jwt, "Content-Type": "application/json" }, // need application/json in here
      body: JSON.stringify({ name_first: firstName, name_last: lastName }),
    };
    fetch("http://localhost:8080/api/v1/admin/user/create", requestOptions)
      .then((res) => {
        console.log("got the res to json!", res);
        return res.json();
      })
      .then((res: CreateUserResponse) => {
        console.log("valid response!", res);
        setDisplayedJWT(res.jwt);
      })
      .catch((err) => console.log("oopsie error", err))
      .finally(() => setAwaitingJWT(false));
    // submit names
  };

  const onClose = () => {
    if (awaitingJWT) return;
    onCloseProp();
  };

  useEffect(() => {
    if (!open) {
      setFirstName("");
      setLastName("");
    }
  }, [open]);

  return (
    <Dialog
      className="fixed inset-0 w-full h-full flex items-center justify-center"
      open={open}
      onClose={onClose}
      initialFocus={cancelButtonRef}
    >
      <Dialog.Overlay className="w-full h-full bg-black absolute  top-0 left-0 opacity-20 z-0" />
      <main className="bg-white z-10 relative w-[416px]">
        <Dialog.Title className="mx-auto w-fit mt-5 text-3xl">
          Add User
        </Dialog.Title>

        <form className="flex flex-col px-16 mt-5">
          <label className="mt-5 w-full">
            First Name
            <input
              type="text"
              className="w-full border px-1 border-black"
              onChange={(e) => setFirstName(e.target.value)}
            />
          </label>
          <label className="mt-5 w-full">
            Last Name
            <input
              type="text"
              className="w-full border px-1 border-black"
              onChange={(e) => setLastName(e.target.value)}
            />
          </label>
          <div className="mx-auto my-5 flex space-x-10 py-5">
            <button
              ref={cancelButtonRef}
              onClick={onClose}
              className="border border-black w-24 hover:font-bold hover:scale-110"
              type="button"
            >
              Cancel
            </button>
            <button
              type="submit"
              onClick={onSubmit}
              className={clsx(
                "border border-black w-24",
                isValid
                  ? "hover:font-bold hover:scale-110"
                  : "pointer-events-none"
              )}
            >
              Submit
            </button>
          </div>
        </form>
        {displayedJWT !== "" && (
          <div className="mb-5 w-full text-center">
            <h4>User's JWT:</h4>
            <p onClick={() => {navigator.clipboard.writeText(displayedJWT)}} className="px-3 w-full break-words">{displayedJWT}</p>
          </div>
        )}
      </main>
    </Dialog>
  );
}
