import React, { useContext } from "react";
import {  googleLogout } from "@react-oauth/google";
import { AppContext } from "../../App";
import SignInButton from "../SignInButton/SignInButton";
import CleanMem from "../mem_reload/MemButton";
import LanguageSelection from "../language/Lang";
const Card = ({
  heading,
  description,
}: {
  heading: string;
  description: string;
}) => {
  return (
    <div className="mx-5 flex-1  mb-0 h-auto flex col-span-12 lg:col-span-4 my-5">
      <div className="card-header flex flex-col w-full">
        <div className="my-2">{heading}</div>
        <div className="rounded-md h-auto w-full  bg-white shadow p-5 flex-1">
          {description}
        </div>
      </div>
    </div>
  );
};

const get_profile = async (token: any) => {
  var requestOptions:RequestInit = {
    method: 'GET',
    redirect: 'follow'
  };
  const res = (await fetch(`https://www.googleapis.com/oauth2/v3/userinfo?access_token=${token}`, requestOptions))
  if(res.status===401) throw new Error("Unauthorized")
  return await res.json()
};

const UserInfo = ({ name, picture, email }: { name: string; picture: string, email:string }) => {
    const {set_user} = useContext(AppContext);
  return (
    <div className="user flex cursor-pointer" onClick={()=>{googleLogout(); set_user(null)}}>
      <img src={picture} className="rounded-full h-10 w-10 mr-2" alt="" />
      <div className="hidden lg:flex lg:flex-col" >
        <div className="name font-medium text-sm">{name}</div>
        <div className="font-thin text-xs">{email}</div>
      </div>
    </div>
  );
};

const Hero = ({
  start_chat,
  chat_started,
}: {
  start_chat: () => void;
  chat_started: boolean;
}) => {
  const { user, set_user } = React.useContext(AppContext);
  const [profile, set_profile] = React.useState<any>(null);
  React.useEffect(()=>{if(user) {
    get_profile(user).then((res) => {
        localStorage.setItem("details", res.sub)
        var myHeaders = new Headers();
        myHeaders.append("Authorization", "Bearer " + user);
        myHeaders.append("Content-Type", "application/json");
        // myHeaders.append("credentials", 'include' );
        var raw = JSON.stringify({
          id : res.sub
        });
        var requestOptions: RequestInit = {
          method: "POST",
          headers: myHeaders,
          body: raw,
          redirect: "follow",
        };
        const response = fetch(process.env.REACT_APP_BACKEND_URL  + "/init", requestOptions)

        console.log(res)
      set_profile(res);
    }).catch(()=>{
        console.log("In catch")
        googleLogout();
        set_user(null)
    });
  }}, [user]);
  return (
    <div className="hero h-full flex flex-col justify-evenly">
      <div className="my-5 mx-8  flex justify-between">
        <div className="font-semibold text-2xl">Sahay</div>
        
        {user ? (
           profile&&<UserInfo name={profile.name} picture={profile.picture} email={profile.email}/>
        ) : (
            <SignInButton />
        )}
      </div>
      <div className="w-full flex flex-col flex-1  mt-10">
        <div className="m-auto">
          <div className="welcome my-5 text-center font-semibold text-5xl">
            Welcome to Sahay!
          </div>
          <div className="tagline mx-auto text-center text-xl my-5">
            Your AI-powered companion for the legal queries
          </div>
          <div className="feature-cards my-16 grid grid-cols-12 lg:w-3/4 mx-auto items-stretch">
            <Card
              heading="🧐 Ask complex questions"
              description="Sahay uses the latest natural language processing technology to provide quick and accurate responses to all your complex legal queries."
            />
            <Card
              heading="🙌 Easy to use"
              description="Sahay is designed to be user-friendly, so you don't need to be a legal expert to use it."
            />
            <Card
              heading="🎨 Personalized Responses"
              description="Sahay provides personalized responses based on the specific details of your legal query."
            />
          </div>
          <div className="info lg:w-3/5 text-center px-5 mx-auto text-sm my-10 ">
            Let's solve your legal querries together. Sahay is powered by AI, so
            surprises and mistakes are possible. Make sure to check the facts,
            and share feedback so we can learn and improve!
          </div>
          <div className="mx-auto w-full flex my-10 text-center justify-center">
            {!chat_started && (
              (user?
              <LanguageSelection
                chat_func={start_chat}
              />
              
              // <button
              //   className="mx-auto bg-black rounded-full text-white py-2 px-8"
              //   onClick={start_chat}
              // >
              //   New Chat
              // </button>
              :
              <SignInButton></SignInButton>
              )
            )}
            {chat_started && (
              (user?<CleanMem></CleanMem>:null
              )
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Hero;
