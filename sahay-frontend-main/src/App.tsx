import React from "react";
import "./App.css";
import Hero from "./components/Hero/Hero";
import Chat from "./components/Chat/Chat";
import { GoogleOAuthProvider } from "@react-oauth/google";

export const AppContext = React.createContext<any>(null);

function App() {
  const [user, set_user] = React.useState(
    localStorage.getItem("user") &&
      (localStorage.getItem("user"))
  );

  const [chat_started, set_chat_started] = React.useState(false);
  return (
    <div className="flex min-h-screen flex-col">
      <AppContext.Provider
        value={{
          user,
          set_user: (u: any) => {
            console.log({u})
            if(!u){
              localStorage.removeItem("user");
              set_user(null);
              return;
            }
            console.log("setting user" ,u)
            localStorage.setItem("user", (u));
            set_user(u);
          },
        }}
      >
        <GoogleOAuthProvider
          clientId={"233687979412-o23j15koep964minh3gouqcoa2s9g3lu.apps.googleusercontent.com" || ""}
        >
          <div className="App flex flex-col bg-cover bg-no-repeat bg-center lg:bg-contain  bg-[url('/src/assets/background.jpg')]">
            <Hero
              chat_started={chat_started}
              start_chat={() => {
                console.log("this is called")
                set_chat_started(true);
              }}
            />
          </div>
          {chat_started && <Chat />}
        </GoogleOAuthProvider>
      </AppContext.Provider>
    </div>
  );
}

export default App;
