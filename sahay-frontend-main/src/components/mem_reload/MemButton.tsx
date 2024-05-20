import React from "react";
const CleanMem = () => {

const handleApiCall = async () => {
    try {
        var myHeaders = new Headers();
        myHeaders.append("Authorization", "Bearer " + localStorage.getItem("user"));
        myHeaders.append("Content-Type", "application/json");
        var raw = JSON.stringify({
          id : localStorage.getItem("details")
        });
        var requestOptions: RequestInit = {
          method: "POST",
          headers: myHeaders,
          body: raw,
          redirect: "follow",
        };
        const response = await fetch(process.env.REACT_APP_BACKEND_URL + "/init", requestOptions); // Replace with your API endpoint
        if (response.ok) {
        const data = await response.json();
        window.location.reload();
        // Assuming the API returns a JSON object with a "message" field
        } 
    } catch (error) {
        console.error('API call error:', error);
    }
    };
    
  return (
    <button
      type="button"
      className="bg-black rounded-full text-white py-2 px-8"
      onClick={()=>{handleApiCall()}}
    >
      Clear Chat History
    </button>
  );
};

export default CleanMem ;
