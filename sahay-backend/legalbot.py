import openai
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI, OpenAIChat
from langchain.chains import  LLMChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.memory import DynamoDBChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from collections import OrderedDict
import os
import json
import boto3
import copy

openai.api_key = ""
os.environ['OPENAI_API_KEY'] = ''
os.environ['AWS_ACCESS_KEY_ID']=''
os.environ['AWS_SECRET_ACCESS_KEY']=''
os.environ['AWS_DEFAULT_REGION']='us-east-1'
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")


dynamodb = boto3.resource('dynamodb')

table = dynamodb.Table('User-details')

first_prompt = """The {ask_for} list contains some things to be asked to the user in a coversation way. You can reformulate the {description} of the field for additional context for forming question
        you should only ask one question at a time even if you don't get all the info.
        Try to ask question in a polite way. Remember that, after asking the question, ask the user if they are unsure about the question. If they are unsure then ask them to type 'Not sure'.
        Don't ask as a list! don't say hi.
        """

desc = {
    'first_name': "This is the first name of the user.",
    'state': "The name of the state in India where someone lives",
    'good_or_service': 'If the grievance is related to a thing or a movable property then it is a good and not a service. If the grievance related to the provision of any facilities such as facilities in connection with banking, financing, insurance, transport, processing, supply of electrical or other energy,  telecom,  boarding  or  lodging  or  both,  housing  construction,  entertainment, amusement or the purveying of news or other information then it might be a service and not a good.',
    'payment_involved': "If any payment/money is paid or any promise has been done to pay money then answer is 'yes' else 'no'",
    'amount_paid': 'If payment is involved then how much amount is already paid and how much is left.',
    'entire_amount_paid' : "If the payment is involved then if the entirety of the sum has already been paid then answer is 'yes' else 'no'", 
    'commercial_or_personal': 'If the consumer query is commercial or personal. Commercial purpose includes use of goods bought by a person and used by them exclusively for the purpose of earning their livelihood, by means of self-employment',
    'date_of_fault_identification': 'Date of identification of fault of the product or service.',
    'filed_complaint': 'If the complaint has already been filed or not.',
    'query_desc': 'Clear description of defect/deficiency in good/service',
    'additional_details': 'Any other additional details about the query, it can be any extra information about the grievance',
}

asking_prompt = PromptTemplate(input_variables=["ask_for", "description"], template=first_prompt)
info_gathering_chain = LLMChain(llm=llm, prompt=asking_prompt)

def check_what_is_empty(user_details):
    ask_for = []
    # Check if fields are empty
    description = ""
    detail_index=0

    key_order = ['first_name', 'state', 'good_or_service', 'payment_involved', 'amount_paid', 'entire_amount_paid', 'commercial_or_personal', 'date_of_fault_identification', 'filed_complaint', 'query_desc', 'additional_details']
    list_of_tuples = [(key, user_details[key]) for key in key_order]
    your_dict = OrderedDict(list_of_tuples)

    for field, value in your_dict.items():
        detail_index+=1
        if value == "" :  # You can add other 'empty' conditions as per your requirements
            ask_for.append(field)
            description = desc[field]
            break
    # return ask_for
    if(len(ask_for) == 0):
        return -1, "", "1"
        # return "Thank you for your patience, I have received all the details. Please wait while I find out the best solution for you."
    else :
        ai_chat = info_gathering_chain.run(ask_for=ask_for, description=description)
        return detail_index, description, ai_chat


def fill_empty_details(table, new_details, id):
    # non_empty_details = {k: v for k, v in new_details.items() if v != ""}
    current_detail = table.get_item(Key = {'SessionId':id})['Item']['user_details']
    done = True
    for field, value in new_details.items():
        if value != "":
            done = False

            current_detail[field] = value
            if field=="payment_involved":
                if (type(value)==bool and not value) or (type(value)==str and value.lower()=="no"):
                    current_detail["amount_paid"] = 0
                    
                    current_detail["entire_amount_paid"] = "Not Applicable"

    table.update_item(Key = {'SessionId':id}, UpdateExpression = 'SET user_details = :val1', ExpressionAttributeValues = {':val1': current_detail})        
    return done

#clarification_chain
clar_prompt = """You will be given a question in {question}. The answer of this question is not sure to the user. 
You need to provide additional information to the user so that they can answer the question. This includes giving some examples
or some additional information which can help the user to answer the question. You can make use of description of the field given in {desc} for some 
relevant information. Give this info and ask the question again to the user."""
clar_chain_prompt = PromptTemplate(
    input_variables=["question", "desc"], template=clar_prompt)
clar_chain = LLMChain(llm=llm, prompt=clar_chain_prompt)

#simplification_chain
simp_query = """
You will be given some details of a query in {query_details}. Your task is to provide a simplified query based on this so that it can be passed to a retriever created from documents to provide a solution to the query.
Remember even though you are summarizing the query, you should not miss out on any important details.
"""
simp_query_prompt = PromptTemplate(
    input_variables=["query_details"], template=simp_query)
simp_query_chain = LLMChain(llm=llm, prompt=simp_query_prompt)

#deduction_prompt
deduction_prompt = """ You just need to extract the following fields from the answer. Do not output anything else
        You will be given an answer. You need to figure out from history and answer, what field is the user talking about and extract those fields\
        Remember values of fields from history.
        If you could not extract about some of the fields from the answer keep those fields as empty, also if some field is unclear then also keep it empty
        The following fields are available to you: 
        first_name: This is the first name of the user."
        state: "The name of the state in India where someone lives"
        good_or_service: If the grievance related to a thing or a movable property including food then it is a good and not a service. If the grievance related to the provision of any facilities such as facilities in connection with banking, financing, insurance, transport, processing, supply of electrical or other energy,  telecom,  boarding  or  lodging  or  both,  housing  construction,  entertainment, amusement or the purveying of news or other information then it might be a service and not a good.
        payment_involved: If any payment/money is paid or any promise has been done to pay money then answer is 'yes' else 'no'.
        amount_paid: If payment_involved is True then how much amount is already paid and how much is left. If no payment involved then zero.
        entire_amount_paid : If the payment is involved then if the entirety of the sum has already been paid then answer is 'yes' else 'no'.
        commercial_or_personal : If the grievance is commercial or personal.
        date_of_fault_identification: Date of identification of fault of the product or service.
        filed_complaint : If the complaint has already been filed then yes else no.
        query_desc : Clear description of defect/deficiency in good/service.
        additional_details : Any other additional details about the query, it can be any extra information about the grievance/query
        If you could not extract about some of the fields from the answer keep those fields as empty
        Do not output anything else just Output only those fields which you could extract in a dictionary format. 
        {chat_history}
        Human : {answer}
    """

#basic_solution_prompt
basic_solution = """You are an expert in law. Use the given question - {question}, context - {context} and chat-history - {chat_history} to propose a 
solution to the query based on the context provided. Remember to describe some laws relevant to this query from context. Mention the solution in step-wise manner.
Keep the answer to about 3 or 4 lines.
Avoid giving any steps which the user has already taken and mentioned in the query.
Do not provide any solutions which are not mentioned in the document or context"""

embeddings = OpenAIEmbeddings()
db = Chroma(persist_directory='chroma_docs/chroma/', embedding_function=embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":5})

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

deduction_chain_prompt = PromptTemplate( input_variables=["chat_history", "answer"], template=deduction_prompt)
deduction_chain = LLMChain(llm=llm, prompt=deduction_chain_prompt)

basic_solution_prompt = PromptTemplate( input_variables=["question", "context", "chat_history"], template=basic_solution)
basic_sol_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    chain_type="stuff",
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": basic_solution_prompt}
)

#initialize memory, deduction and basic solution chain
def init_mem(id: str):
    global memory
    global basic_sol_chain
    global deduction_chain
    global table
    # global user_details
    # receiving_details = True
    # global receiving_details

    # Create the DynamoDB table.
    try:
        table1 = dynamodb.create_table(
            TableName="conversation-store",
            KeySchema=[{"AttributeName": "SessionId", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "SessionId", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        # Wait until the table exists.
        table1.meta.client.get_waiter("table_exists").wait(TableName="conversation-store")
    except:
        pass
    
    user_details = {'first_name' : "",
                'state':"",
                'good_or_service':"",
                'payment_involved':"",
                'amount_paid':"",
                'entire_amount_paid' : "", 
                'commercial_or_personal':"",
                'date_of_fault_identification':"",
                'filed_complaint':"",
                'query_desc':"",
                'additional_details':""}

    key_order = ['first_name', 'state', 'good_or_service', 'payment_involved', 'amount_paid', 'entire_amount_paid', 'commercial_or_personal', 'date_of_fault_identification', 'filed_complaint', 'query_desc', 'additional_details']
    list_of_tuples = [(key, user_details[key]) for key in key_order]
    your_dict = OrderedDict(list_of_tuples)

    table.put_item(Item={'receiving_details' : True, 'user_details' : your_dict, 'detail_index' : 0, 'obj_desc' : "", 'SessionId' : id})
    
    # for val in user_details.values():
    #     val = ""
    chat_history = DynamoDBChatMessageHistory(table_name='conversation-store', session_id=id)
    chat_history.clear()
    memory.chat_memory=chat_history
    
    memory.chat_memory.add_ai_message("Hi! I am Sahay. How may I help You?")

    # memory.chat_memory.clear()
    print(table.get_item(Key = {'SessionId':id})['Item']['user_details'])

    deduction_chain.memory = memory
    basic_sol_chain.memory = memory
 
    


def get_dict(s) :
    i = s.find("{")
    j = s.find("}")
    return json.loads(s[i:j+1])


# receiving_details = True
# obj_desc = ""
# detail_index = 0
def chat_qa(answer, id) :
    # answer = input("User: ")
    # global receiving_details
    # global obj_desc
    # global detail_index
    obj_desc = table.get_item(Key = {'SessionId':id})['Item']['obj_desc']
    detail_index = table.get_item(Key = {'SessionId':id})['Item']['detail_index']
    if (table.get_item(Key = {'SessionId':id})['Item']['receiving_details']) :
    # what if user is unsure, we should run a different chain for clarification
        if(answer == "Not sure") :
            ques_unsure = memory.chat_memory.messages[-1]
            clar_output = clar_chain.run(question=ques_unsure, desc=obj_desc)
            memory.chat_memory.add_ai_message(clar_output)
            return str(clar_output)
            # print(f"Bot : {clar_output}")
            # continue
        
        #name/state
        if (detail_index==1):
            fill_empty_details(table, {'first_name': answer}, id)
        elif (detail_index==2):
            fill_empty_details(table, {'state': answer}, id)
        elif (detail_index==11):
            fill_empty_details(table, {'additional_details': answer}, id)
        else:
            question = str(memory.chat_memory.messages[-1])
            if (detail_index!=0):
                answer=question+answer
            output = deduction_chain.run(answer=answer)
            try : 
                form_output =  get_dict(output)
            except :
                form_output = {}
            # print(f"form output - {form_output}")
            fill_empty_details(table, form_output, id) # updating new details
        
        # print(f"user details - {user_details}")
        index, desc, question = check_what_is_empty(table.get_item(Key = {'SessionId':id})['Item']['user_details']) # asking question for empty fields
        table.update_item(Key = {'SessionId':id}, UpdateExpression = 'SET detail_index = :val1', ExpressionAttributeValues = {':val1': index})
        table.update_item(Key = {'SessionId':id}, UpdateExpression = 'SET obj_desc = :val1', ExpressionAttributeValues = {':val1': desc})
        print(detail_index)
        if(question == "1"):
            # receiving_details = False
            table.update_item(Key = {'SessionId':id}, UpdateExpression = 'SET receiving_details = :val1', ExpressionAttributeValues = {':val1': False})
            query_details = table.get_item(Key = {'SessionId':id})['Item']['user_details']
            if (type(query_details["commercial_or_personal"])==str and "commercial" in query_details["commercial_or_personal"]):
             return "This matter will not fall within the ambit of CPA"
            if ((type(query_details["payment_involved"])==bool and not query_details["payment_involved"]) or (type(query_details["payment_involved"])==str and "no" in query_details["payment_involved"].lower())):
             return "This matter will not fall within the ambit of CPA"
            query_details.pop('first_name')
            new_query = simp_query_chain.run(query_details=query_details)
            output = basic_sol_chain({'question' : new_query})
            return "Thank you for your patience. " + str(output['answer'])
            # return "Thank you for your patience, I have received all the details. Please wait while I find out the best solution for you."
            # print("Thank you for your patience, I have received all the details. Please wait while I find out the best solution for you.")
            # break
        
        #rulebased modification of questions based on good or service
        #date of fault identification for the first time
        if (detail_index==8):
            if ("good" in table.get_item(Key = {'SessionId':id})['Item']['user_details']['good_or_service'].lower()):
                question = "Can you please provide the date when you first identified the fault in your product? If you're unsure about the exact date, please type 'Not sure'."
            if ("service" in table.get_item(Key = {'SessionId':id})['Item']['user_details']['good_or_service'].lower()):
                question = "Can you please provide the date when you first identified the fault in the service? If you're unsure about the exact date, please type 'Not sure'."
        #clear description of grievance
        if (detail_index==10):
            if ("good" in table.get_item(Key = {'SessionId':id})['Item']['user_details']['good_or_service'].lower()):
                question = "Could you please provide a clear description of the defect/fault in the good? If you are unsure about the question, please type 'Not sure'."
            if ("service" in table.get_item(Key = {'SessionId':id})['Item']['user_details']['good_or_service'].lower()):
                question = "Could you please provide a clear description of the deficiency/fault in the service? If you are unsure about the question, please type 'Not sure'."
    
        memory.chat_memory.add_ai_message(question)
        return str(question)
        # print(f"Bot : {question}")
        if output is None:
            return "I am sorry I did not understand that. Can you please rephrase it?"
            # print("Bot: I am sorry I did not understand that. Can you please rephrase it?")
    else :
        query_details = table.get_item(Key = {'SessionId':id})['Item']['user_details']
        query_details.pop('first_name')
        query_details['additional_details'] += ". " + answer
        new_query = simp_query_chain.run(query_details=query_details)
        output = basic_sol_chain({'question' : new_query})
        return str(output['answer'])
        # print(output['answer'])

    # print(f"{response}")
