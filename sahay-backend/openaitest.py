import openai
import gradio as gr

openai.api_key = "" 
from llama_index import GPTVectorStoreIndex, Document, SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_disk

from llama_index.agent import OpenAIAgent, ReActAgent
from llama_index.llms import OpenAI
import os

os.environ['OPENAI_API_KEY'] = '' 


#
#messages = [
 #   {"role": "system", "content": "You are an AI specialized in Food. Do not answer anything other than food-related queries."},
#]
messages = [
    {"role": "system", "content": "You are an AI specialized in law. Do not answer anything other than law-related queries."},
 {"role": "user", "content": "I have complaint against Amazon."},
        {"role": "assistant", "content": "Have you purchased anything recently?"},


]


documents = SimpleDirectoryReader('data/').load_data()
index = GPTVectorStoreIndex.from_documents(documents)
# Save your index to a index.json file
# index.save_to_disk('index.json')
# # Load the index from your saved index.json file
# index = GPTVectorStoreIndex.load_from_disk('index.json')


index.storage_context.persist(persist_dir="./storage")
index = load_index_from_disk(StorageContext.from_defaults(persist_dir="./storage"))

#response = index.query("i have a complaint")
#print(response)

def chatbot(input):
    if input:
        response= index.query(input)
        print(response)
        return response

inputs = gr.inputs.Textbox(lines=7, label="Chat with AI")
outputs = gr.outputs.Textbox(label="Reply")

gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="Sahay",
            description="Ask anything you want",
            theme="compact").launch(share=True)
