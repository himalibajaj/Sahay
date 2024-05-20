import React from "react";
import { useEffect } from "react";
import loading_icon from "../../assets/typing.svg";


var welcome_msg:string = "";
if (localStorage.getItem("welcome")){
  welcome_msg = localStorage.getItem("welcome") as string
} ;

const MessageBox = ({
  send_message,
  response_pending,
}: {
  send_message: (msg: string) => void;
  response_pending: boolean;
}) => {
  const [message, set_message] = React.useState("");
  return (
    <div className="message-box w-full flex mx-auto  my-5 sticky bottom-5 mt-auto">
      <div className="flex flex-1">
        {/* <button>New chat</button> */}
        <form
          className="flex flex-1"
          action=""
          onSubmit={(e) => {
            e.preventDefault();
            send_message(message);
            set_message("");
          }}
        >
          <input
            disabled={response_pending}
            onChange={(e) => {
              set_message(e.target.value);
            }}
            value={message}
            type="text"
            className="appearance-none mr-3 px-3.5 !outline-none block w-full rounded-md border-0  py-2 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-blue-600 sm:text-sm sm:leading-6"
          />
          <button
            disabled={message === ""}
            className="bg-black rounded h-10 w-10 text-white disabled:bg-gray-700 "
          >
            <i className="fa-solid fa-paper-plane"></i>
          </button>
        </form>
      </div>
    </div>
  );
};

enum From {
  sahay,
  user,
}

const Message = ({ from, text }: { from: From; text: string }) => {
  return (
    <div
      className={
        "flex message py-2 my-2 px-3 text-sm " +
        (from === From.sahay
          ? "rounded-t-lg rounded-r-lg bg-gray-50 mr-auto "
          : "rounded-t-lg rounded-l-lg bg-blue-500 text-white ml-auto")
      }
    >
      {text}
    </div>
  );
};

const TypingMessage = () => {
    return (
      <div
        className={
          "flex message py-0 my-2 px-3 text-sm rounded-t-lg rounded-r-lg bg-gray-50 mr-auto "
        }
      >

        <img src={loading_icon} className="h-10" alt="" />
      </div>
    );
  };

const Messages = ({
  messages,
  show_typing
}: {
  messages: { from: From; text: string }[];
  show_typing: boolean
}) => {
  const lastMessage = React.useRef<HTMLDivElement>(null);
  useEffect(() => {
    lastMessage.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);
  return (
    <div className="messages flex flex-col" style={{ whiteSpace: 'pre-line' }}>
      
      <Message from={From.sahay} text={welcome_msg} />
      {messages.map((m) => (
        <Message from={m.from} text={m.text} />
      ))}
      {show_typing&&<TypingMessage/>}
      <div className="h-5" ref={lastMessage}></div>
    </div>
  );
};

const send_message = async (messages: { from: From; text: string }[]) => {
  var myHeaders = new Headers();
  myHeaders.append("Authorization", "Bearer " + localStorage.getItem("user"));
  myHeaders.append("Content-Type", "application/json");
  // myHeaders.append("credentials", 'include' );
  // console.log("sent by " + localStorage.getItem("details"))
  var raw = JSON.stringify({
    // messages: messages.map((m) => {
    //   return {
    //     role: m.from === From.sahay ? "assistant" : "user",
    //     content: m.text,
    //   };
    // }),
    input: messages[messages.length - 1].text,
    id : localStorage.getItem("details"),
    lang : localStorage.getItem("language")
  });

  var requestOptions: RequestInit = {
    method: "POST",
    headers: myHeaders,
    body: raw,
    redirect: "follow",
  };

  var res =  await (
    await fetch(process.env.REACT_APP_BACKEND_URL + "/message", requestOptions)
  ).json();
  return res
}

const Chat = () => {
  const [messages, set_messages] = React.useState<
    { from: From; text: string }[]
  >([]);
  const [response_pending, set_response_pending] = React.useState(false);
  return (
    <>
      <div className="chat  flex flex-1 flex-col lg:w-3/4 px-5 lg:mx-auto">
        <Messages messages={messages} show_typing = {response_pending}/>
        <MessageBox
          response_pending={response_pending}
          send_message={async (text: string) => {
            set_response_pending(true);
            set_messages([...messages, { from: From.user, text }]);
            set_messages([
              ...messages,
              { from: From.user, text },
              {
                from: From.sahay,
                text: (
                  await send_message([...messages, { from: From.user, text }])
                ),
              },
            ]);
            set_response_pending(false);
          }}
        />
      </div>
    </>
  );
};

export default Chat;
