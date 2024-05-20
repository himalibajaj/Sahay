import openai
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI, OpenAIChat
from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain, create_tagging_chain, create_tagging_chain_pydantic
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
# from enum import Enum
# from pydantic import BaseModel, Field
import os
import json


openai.api_key = ""
os.environ['OPENAI_API_KEY'] = ''

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

user_details = {'first_name' : "",
                'state':"",
                'Good_or_service':"",
                'payment_involved':"",
                'amount_paid':"",
                'Commercial_or_Personal':"",
                'date_of_fault_identification':"",
                'filed_complaint':"",
                'query_desc':"",
                'additional_details':""}

first_prompt = """The {ask_for} list contains some things to be asked to the user in a coversation way. You can reformulate the {description} of the field for additional context for forming question
        you should only ask one question at a time even if you don't get all the info.
        Try to ask question in a polite way. Remember that, after asking the question, ask the user if they are unsure about the question. If they are unsure then ask them to type 'Not sure'.
        Don't ask as a list! don't say hi.
        """

desc = {
    'first_name': "This is the first name of the user.",
    'state': "The name of the state in India where someone lives",
    'Good_or_service': 'If the grievance is related to a thing or a movable property then it is a good and not a service. If the grievance related to the provision of any facilities such as facilities in connection with banking, financing, insurance, transport, processing, supply of electrical or other energy,  telecom,  boarding  or  lodging  or  both,  housing  construction,  entertainment, amusement or the purveying of news or other information then it might be a service and not a good.',
    'payment_involved': "If any payment/money is paid or any promise has been done to pay money then answer is yes else no",
    'amount_paid': 'If payment is involved then how much amount is already paid and how much is left. If no payment involved then zero.',
    'Commercial_or_Personal': 'If the consumer query is commercial or personal. Commercial purpose includes use of goods bought by a person and used by them exclusively for the purpose of earning their livelihood, by means of self-employment',
    'date_of_fault_identification': 'Date of identification of fault of the product or service.',
    'filed_complaint': 'If the complaint has already been filed or not.',
    'query_desc': 'Clear description of defect/deficiency in good/service',
    'additional_details': 'Any other additional details about the query, it can be any extra information about the grievance/query',
}

# imperfection or shortcoming in the quality, quantity, potency, purity or standard of the good 
asking_prompt = PromptTemplate(input_variables=["ask_for", "description"], template=first_prompt)
info_gathering_chain = LLMChain(llm=llm, prompt=asking_prompt)

def check_what_is_empty(user_details):
    ask_for = []
    # Check if fields are empty
    description = ""
    for field, value in user_details.items():
        if value == "" :  # You can add other 'empty' conditions as per your requirements
            ask_for.append(field)
            description = desc[field]
            break
    # return ask_for
    if(len(ask_for) == 0):
        return "", 1
        # return "Thank you for your patience, I have received all the details. Please wait while I find out the best solution for you."
    else :
        ai_chat = info_gathering_chain.run(ask_for=ask_for, description=description)
        return description, ai_chat


def fill_empty_details(current_detail, new_details):
    # non_empty_details = {k: v for k, v in new_details.items() if v != ""}
    done = True
    for field, value in new_details.items():
        if value != "":
            done = False
            current_detail[field] = value
            if field=="payment_involved" and value.lower()=="no":
                current_detail["amount_paid"] = 0
    return done
memory = ConversationBufferMemory(memory_key="history")

deduction_prompt = """ You just need to extract the following fields from the answer. Do not output anything else
        You will be given an answer. You need to figure out from history and answer, what field is the user talking about and extract those fields\
        Remember values of fields from history.
        If you could not extract about some of the fields from the answer keep those fields as empty, also if some field is unclear then also keep it empty
        The following fields are available to you: 
        first_name: This is the first name of the user."
        state: "The name of the state in India where someone lives"
        Good_or_service: If the grievance related to a thing or a movable property including food then it is a good and not a service. If the grievance related to the provision of any facilities such as facilities in connection with banking, financing, insurance, transport, processing, supply of electrical or other energy,  telecom,  boarding  or  lodging  or  both,  housing  construction,  entertainment, amusement or the purveying of news or other information then it might be a service and not a good.
        payment_involved: 
        amount_paid: If payment_involved is True then how much amount is already paid and how much is left. If no payment involved then zero.
        Commercial_or_Personal : If the grievance is commercial or personal.
        date_of_fault_identification: Date of identification of fault of the product or service.
        filed_complaint : If the complaint has already been filed then yes else no.
        query_desc : Clear description of defect/deficiency in good/service.
        additional_details : Any other additional details about the query, it can be any extra information about the grievance/query
        If you could not extract about some of the fields from the answer keep those fields as empty
        Do not output anything else just Output only those fields which you could extract in a dictionary format. 
        {history}
        Human : {answer}
    """

prompt = PromptTemplate(
    input_variables=["history", "answer"], template=deduction_prompt)
# "If any payment/money is paid or any promise has been done to pay money then yes otherwise no"
# And pick one field which you could not retrieve from answer and ask back a question to the user to get that field value
deduction_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
# memory.chat_memory.add_ai_message("Did you file a complaint")


clar_prompt = """You will be given a question in {question}. The answer of this question is not sure to the user. 
You need to provide additional information to the user so that they can answer the question. This includes giving some examples
or some additional information which can help the user to answer the question. You can make use of description of the field given in {desc} for some 
relevant information. Give this info and ask the question again to the user."""
clar_chain_prompt = PromptTemplate(
    input_variables=["question", "desc"], template=clar_prompt)

clar_chain = LLMChain(llm=llm, prompt=clar_chain_prompt)

def get_dict(s) :
    i = s.find("{")
    j = s.find("}")
    return json.loads(s[i:j+1])


obj_desc = ""
while True :
    answer = input("User: ")

    # what if user is unsure, we should run a different chain for clarification
    if(answer == "Not sure") :
        ques_unsure = memory.chat_memory.messages[-1]
        clar_output = clar_chain.run(question=ques_unsure, desc=obj_desc)
        memory.chat_memory.add_ai_message(clar_output)
        print(f"Bot : {clar_output}")
        continue

    output = deduction_chain.run(answer=answer)
    try : 
       form_output =  get_dict(output)
    except :
        form_output = {}
    # print(f"form output - {form_output}")
    fill_empty_details(user_details, form_output) # updating new details
    # print(f"user details - {user_details}")
    obj_desc, question = check_what_is_empty(user_details) # asking question for empty fields
    if(question == 1):
        print("Thank you for your patience, I have received all the details. Please wait while I find out the best solution for you.")
        break
    memory.chat_memory.add_ai_message(question)
    print(f"Bot : {question}")
    if output is None:
        print("Bot: I am sorry I did not understand that. Can you please rephrase it?")



# query_details = {
#                 'state':"Delhi",
#                 'Good_or_service':"Good",
#                 'payment_involved':"Yes",
#                 'amount_paid':"1200 rupees",
#                 'Commercial_or_Personal':"Personal",
#                 'date_of_fault_identification':"20th August 2023",
#                 'filed_complaint':"No",
#                 'query_desc':"Amazon sent a defective product",
#                 'additional_details':"I returned the product but they are not refunding the money"}

query_details = user_details
print(query_details)
cpa = True
if query_details["Commercial_or_Personal"].lower()=="commercial" or query_details["payment_involved"].lower()=="no":
    cpa = False
query_details.pop('first_name')
simp_query = """
You will be given some details of a query in {query_details}. Your task is to provide a simplified query based on this so that it can be passed to a retriever created from documents to provide a solution to the query.
Remember even though you are summarizing the query, you should not miss out on any important details.
"""
simp_query_prompt = PromptTemplate(
    input_variables=["query_details"], template=simp_query)
simp_query_chain = LLMChain(llm=llm, prompt=simp_query_prompt)
new_query = simp_query_chain.run(query_details=query_details)
# print(simp_query_chain.run(query_details=query_details))


embeddings = OpenAIEmbeddings()
db = Chroma(persist_directory='chroma_docs/chroma/', embedding_function=embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":5})

memory_ret = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

basic_solution = """Use the given question - {question}, context - {context} and chat-history - {chat_history} to propose a 
detailed solution to the query based on the documents provided. Mention the solution in step-wise manner.
Avoid giving any steps which the user has already taken and mentioned in the query.
Also, provide some things about which are preventing you from providing a specific solution and need more information from the user.
Do not provide any solutions which are not mentioned in the document or context"""
basic_solution_prompt = PromptTemplate(
    input_variables=["question", "context", "chat_history"], template=basic_solution)

basic_sol_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    # llm=ChatOpenAI(),
    chain_type="stuff",
    retriever=retriever,
    memory=memory_ret,
    combine_docs_chain_kwargs={"prompt": basic_solution_prompt}
)


# ques_sol= """Use the given question - {question}, context - {context} and chat-history - {chat_history} to think of a 
# detailed solution to the query based on the documents provided. But, don't output this solution. 
# Output should be one question regarding some things which are preventing you from providing the absolute best solution and need more information from the user.
# Do not make-up any questions. Only ask questions which are relevant with respect to the context and those which are not already mentioned in the query.
# """
# ques_sol_prompt = PromptTemplate(
#     input_variables=["question", "context", "chat_history"], template=ques_sol)

# ques_sol_chain = ConversationalRetrievalChain.from_llm(
#     llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
#     # llm=ChatOpenAI(),
#     chain_type="stuff",
#     retriever=retriever,
#     memory=memory_ret,
#     combine_docs_chain_kwargs={"prompt": ques_sol_prompt}
# )


# question = "I have a query regarding a defective pillow sent by Amazon. I live in Delhi and it was a personal purchase. I paid 1200 rupees for the product. The fault was identified on 20th August 2023. I have not filed a complaint yet. I returned the product, but Amazon is not refunding the money."
# question = "I have a query regarding a defective pillow sent by Amazon. I live in Delhi and it was a personal purchase. I paid 1200 rupees for the product. The fault was identified on 20th August 2023. I have not filed a complaint yet. I returned the product, but Amazon is not refunding the money."
# question = "List down some consumer laws that are violated if recieved a defective product from a company(jurisdiction is state) in India  and possible solutions for redressal. Also can you provide some specific people or court to contact."

init = False
while cpa :
    if(init == True) : 
        new_query = input("User: ")
    init = True
    output = basic_sol_chain({'question' : new_query})
    print(output['answer'])

if not cpa :
    print("This matter will not fall within the ambit of CPA")

# from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
# from langchain.schema.messages import SystemMessage
# from langchain.prompts import MessagesPlaceholder
# from langchain.agents import AgentExecutor
# from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
# from langchain.agents.agent_toolkits import create_retriever_tool
# memory_key = "history"
# memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)

# tool = create_retriever_tool(
#     retriever,
#     "search_consumer_law_docs",
#     "Searches and returns documents regarding consumer laws in India. You should use this tool when there is some consumer query and you to need to provide a solution based on the documents.",
# )
# tools = [tool]
# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
# message = SystemMessage(
#     content=(
#         """Use the given question - {question}, context - {context} and chat-history - {chat_history} to propose a 
# detailed solution to the query based on the documents provided. Mention the solution in step-wise manner.
# Avoid giving any steps which the user has already taken and mentioned in the query.
# Also, provide some things about which are preventing you from providing a specific solution and need more information from the user.
# Do not provide any solutions which are not mentioned in the document or {context}"""
#     )
# )
# prompt = OpenAIFunctionsAgent.create_prompt(
#     system_message=message,
#     extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
# )
# agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
# agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True,
#                                    return_intermediate_steps=True)
# question = "I have a query regarding a defective pillow sent by Amazon. I live in Delhi and it was a personal purchase. I paid 1200 rupees for the product. The fault was identified on 20th August 2023. I have not filed a complaint yet. I returned the product, but Amazon is not refunding the money."

# result = agent_executor({"input": question})

# question_corr = """You will be given some query and it's details in {query} and you will be given a question in {question}. You need to figure out if the answer of this question is present within the query. If it is then output 'Yes' else 'No'"""
# quo_corr_prompt = PromptTemplate(
#     input_variables=["query", "question"], template=question_corr)
# quo_chain = LLMChain(llm=llm, prompt=quo_corr_prompt) 
# print(quo_chain.run(query=question, question=output['answer']))


# print(output['source_documents'])
