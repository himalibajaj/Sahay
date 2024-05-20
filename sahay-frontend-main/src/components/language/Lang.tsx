import React, { useState } from 'react';
import { AppContext } from "../../App";
// Assuming you have a function startChat that triggers a new chat with the selected language
// const startChat = (selectedLanguage: string) => {
//   // Function to start a new chat with the selected language
//   console.log(`Starting new chat in ${selectedLanguage}...`);
//   // Add your logic here to start a chat with the selected language
// };

const LanguageSelection = ({
    chat_func
  }: {
    chat_func: () => void;
    
  }) => {
  const [selectedLanguage, setSelectedLanguage] = useState<string | null>(null);

  const HandleLanguageChange = async (event: React.ChangeEvent<HTMLSelectElement>) => {
    // const { user, set_user } = React.useContext(AppContext);
    setSelectedLanguage(event.target.value);
    localStorage.setItem("language", event.target.value);
    var myHeaders = new Headers();
    myHeaders.append("Authorization", "Bearer " + localStorage.getItem("user"));
    myHeaders.append("Content-Type", "application/json");
    // myHeaders.append("credentials", 'include' );
    var raw = JSON.stringify({
        lang : event.target.value
    });
    var requestOptions: RequestInit = {
        method: "POST",
        headers: myHeaders,
        body: raw,
        redirect: "follow",
    };
    var res =  await (
      await fetch(process.env.REACT_APP_BACKEND_URL  + "/setlang", requestOptions)
    ).json();
    console.log(res)
    localStorage.setItem("welcome", res)
    // const response = fetch(process.env.REACT_APP_BACKEND_URL  + "/setlang", requestOptions)
};

  return (
    <div>
        {
            !selectedLanguage &&
            <select
                value={selectedLanguage || ''}
                onChange={HandleLanguageChange}
                className="bg-black rounded-full text-white py-2 px-8"
            >
                <option value="" disabled>
                Select Language
                </option>
                <option value="english">English</option>
                <option value="hindi">Hindi</option>
                <option value="marathi">Marathi</option>
                <option value="bengali">Bengali</option>
                <option value="tamil">Tamil</option>
                <option value="punjabi">Punjabi</option>
                {/* Add other languages as needed */}
            </select>
        }
      { 
        selectedLanguage &&
        <button
            onClick={chat_func}
            disabled={!selectedLanguage}
            className="mx-auto bg-black rounded-full text-white py-2 px-8 mt-2"
        >
            New Chat
        </button>
        }
    </div>
  );
};

export default LanguageSelection;
