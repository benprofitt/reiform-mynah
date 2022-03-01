import React, { useState } from "react";
import Cookies from "universal-cookie";

export const authCookieName = "reiform-api-key-jwt";
export const authCookieOptions = { path: '/' }

export default function LoginPage(): JSX.Element {
  const cookies = new Cookies();

  const [jwt, setJwt] = useState("");

  const onSubmit = () => {
    cookies.set(authCookieName, jwt, authCookieOptions);
  };

  return (
    <div className="flex items-center justify-center mt-20">
      <form onSubmit={onSubmit}>
        <input
          className="h-10 w-80 border rounded border-black px-4"
          placeholder="Enter JWT Here..."
          value={jwt}
          onChange={(e) => setJwt(e.target.value)}
        />
        <button
          type="submit"
          className="ml-2 text-center border rounded border-black h-10 w-20 hover:border-2"
        >
          Submit!
        </button>
      </form>
    </div>
  );
}
