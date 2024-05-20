import { useGoogleLogin } from "@react-oauth/google";
import { AppContext } from "../../App";
import React from "react";

const SignInButton = () => {
  const { set_user } = React.useContext(AppContext);
  const login = useGoogleLogin({
    onSuccess: (codeResponse) => {
      console.log({ codeResponse });
      set_user(codeResponse.access_token);
    },
  })
  return (
    <button
      type="button"
      className="bg-black rounded-full text-white py-2 px-8"
      onClick={()=>login()}
    >
      Sign in
    </button>
  );
};

export default SignInButton;
