import { Dialog } from "@headlessui/react";
import clsx from "clsx";
import React, { useEffect, useRef, useState } from "react";
import makeRequest from "../utils/apiFetch";
import { CreateUserResponse } from "../utils/types";

export interface AddUserModalProps {
  open: boolean;
  onClose: () => void;
}

export default function AddUserModal(props: AddUserModalProps): JSX.Element {
  const { open, onClose: onCloseProp } = props;
  const cancelButtonRef = useRef(null);
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [displayedJWT, setDisplayedJWT] = useState("");
  const [awaitingJWT, setAwaitingJWT] = useState(false);

  const isValid = Boolean(firstName) && Boolean(lastName);

  const onSubmit = (e: React.MouseEvent<HTMLButtonElement, MouseEvent>) => {
    e.preventDefault();
    if (!isValid || awaitingJWT) return;
    setAwaitingJWT(true);
    makeRequest<CreateUserResponse>(
      "POST",
      { name_first: firstName, name_last: lastName },
      "/api/v1/admin/user/create",
      'application/json'
    )
      .then((res) => {
        // console.log("valid response!", res);
        setDisplayedJWT(res.jwt);
      })
      .catch((_err) => {
        // console.log("oopsie error", _err)
      })
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
            <p
              onClick={() => {
                navigator.clipboard.writeText(displayedJWT);
              }}
              className="px-3 w-full break-words"
            >
              {displayedJWT}
            </p>
          </div>
        )}
      </main>
    </Dialog>
  );
}
