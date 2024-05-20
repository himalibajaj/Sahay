import openai
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
import os
import json

openai.api_key = ""
os.environ['OPENAI_API_KEY'] = ''


template = """
Assistant will be provided with customer service query.
Assistant needs to extract the following details from the query.
1. User details: 
    Name, 
    State of Residence, 
2. Query details: 
    Clear and concise description of the issue or query, 
    Relevant product or service details (product name, model number, service type, etc.),
    Date of purchase. 
3. Transaction Details: 
    Order number (if applicable), 
    Payment details (transaction ID, mode of payment), 

Assistant can ask for atmost 2 details in one question.
If the user doesn't provide some of the above details, assistant needs to ask clarifying follow-up questions
so that it can extract all the information from the user. Also proceed to ask questions, in a sequential order\
User details then Query details and finally Transaction details.
Do not ask for redundant questions.
Continue asking clarifying questions until you obtain all the details specified above.
Once all details are recieved, output all the details in a python dictionary format, in this case make sure to start the answer with "Done".

{history}
Human : {human_input}
Assistant :
"""

# template_doc_retrieval = """
# Assistant will be provided with {context} which contains the details of the user.

# """
prompt = PromptTemplate(
    input_variables=["history", "human_input"], template=template)

memory = ConversationBufferMemory(memory_key="history")
chatgpt_chain = LLMChain(
    llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
    prompt=prompt,
    verbose=False,
    memory=memory,
)

# document intialisation part starts below
template_doc = """Use the following pieces of chat_history to answer the question at the end. If you are unsure about an answer, ask questions to the user to get more details and then use that context to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{chat_history} {context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["question", "context", "chat_history"], template=template_doc,)

# load documents
# loaded_file = 'data/CP_CDRC_AmendmentRules2022.pdf'
# loader = PyPDFLoader(loaded_file)
# documents = loader.load()
# # split documents
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000, chunk_overlap=150)
# docs = text_splitter.split_documents(documents)
# # define embedding
# embeddings = OpenAIEmbeddings()
# # create vector database from data
# db = DocArrayInMemorySearch.from_documents(docs, embeddings)


embeddings = OpenAIEmbeddings()
db = Chroma(persist_directory='chroma_docs/chroma/', embedding_function=embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
# create a chatbot chain. Memory is managed externally.
# chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}


memory_ret = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    chain_type="stuff",
    retriever=retriever,
    memory=memory_ret,
    combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
    # chain_type_kwargs=chain_type_kwargs,
    # return_source_documents=True,
    # return_generated_question=True,
    # prompt= QA_CHAIN_PROMPT,
)

# loop to gather all the details from the user
test_string = ''' '''
# receiving_details = True
# while True:
#     text_input = input("User: ")
#     if(receiving_details) :
#         response = chatgpt_chain.run(human_input = text_input)
#         if(response[:4] == "Done"):
#             test_string = response[5:]
#             # print(response[5:])
#             print(f"Assistant : I have recieved all the details, how can I assist you further ?")
#             receiving_details = False
#             memory_ret.chat_memory.add_user_message(test_string)
#             # break
#     else :
#         result = qa({"question": text_input})
#         response = result['answer']

#     print(f"{response}")

receiving_details = True

def chat_qa(text_input) :
    global receiving_details
    if(receiving_details) :
        response = chatgpt_chain.run(human_input = text_input)
        if(response[:4] == "Done"):
            test_string = response[5:]
            # print(response[5:])
            receiving_details = False
            memory_ret.chat_memory.add_user_message(test_string)
            custom_ques = "Propose a solution for the issue of the user based on the document provided."
            result = qa({"question": custom_ques })
            response = result['answer']
            return str(response)
            # return "Assistant : I have recieved all the details, how can I assist you further ?"
            
            # break
    else :
        result = qa({"question": text_input})
        response = result['answer']

    return str(response)
    # print(f"{response}")








































    

# loop to ask questions about document

# while True:
#     query = input("User: ")
#     # chat_history = []
#     # qa = load_db(loaded_file,"stuff", 4)
#     result = qa({"question": query})
#     # chat_history.extend([(query, result["answer"])])
#     # db_query = result["generated_question"]
#     # db_response = result["source_documents"]
#     answer = result['answer']
#     # print(f"Assistant : {db_query}")
#     # print(f"Assistant : {db_response}")
#     print(f"Assistant : {answer}")

# res = json.loads(output)


# test_string = '''{
#   "User details": {
#     "Name": "kush",
#     "Location": {
#       "City": "Indore",
#       "State": "Madhya Pradesh"
#     }
#   },
#   "Query details": {
#     "Issue or Query": "The phone is not switching on",
#     "Product or Service details": {
#       "Product name": "Samsung Galaxy S21"
#     },
#     "Date of purchase": "20th August 2023"
#   },
#   "Transaction details": {
#     "Order number": "3452"
#   }
# }'''
# context = json.loads(test_string)


# print("context done")


# res contains the details in a dictionary format
# print(res["User details"]["Location"]["City"])




# # Build prompt
# template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
# {context}
# Question: {query}
# Helpful Answer:"""
# QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "query"],template=template,)

# # Run chain
# from langchain.chains import RetrievalQA
# # question = "Is probability a class topic?"
# qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name= "gpt-3.5-turbo", temperature=0),
#                                        retriever=retriever,
#                                        return_source_documents=True,
#                                        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

# while True:
#     question = input("User: ")
#     result = qa_chain({"query": question, "context" : context})
#     answer = result["result"]
#     print(f"Assistant : {answer}")




# return qa

# embedding = OpenAIEmbeddings()
# vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# print(output)
