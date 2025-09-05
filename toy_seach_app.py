import streamlit as st
import pandas as pd 
import numpy as np
import boto3
import yaml
import json

from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

st.title('App to Search for Toys')


# load in environment variables 
with open('config.yaml') as info:
  env = yaml.load(info, Loader=yaml.Loader)

# connect to pinecone
pc = Pinecone(api_key=env['pinecone_api_key'])

#connect to openai for document embeddings
embeddings = OpenAIEmbeddings(api_key=env['open_ai_api_key'])

# set up vector store to upsert to index and then embedd the documents using openAI Embedding from above
index = pc.Index('product-descriptions')
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# connect to  amazon bedrock
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

def get_products_information(question, embeddings, vector_store, bedrock):

    # create initial prompt template for llm
    prompt = """You are a friendly chatbot capable of answering questions related to products. User's can ask questions about its description,
       and prices. Be polite and redirect conversation specifically to product information in the Context. If you dont know the answer say I don't know

        Question: {question}
       
       Context: {context} 
    """

    # using the question asked get 5 closest documents
    results = vector_store.similarity_search(question, k=5)


    # loop through results and append document information and metadata
    information = []
    metadata = []
    for res in results:
        information.append(f"* {res.page_content + f" The price of the product is {res.metadata['list_price']}"} [{res.metadata}]")
        metadata.append(res.metadata)

    
    
    # fill in returned information from pinecone vector store into prompt
    prompt = prompt.format(context=" ".join(information), question=question)

    
    # create payload for model
    payload = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount":3000,
            "stopSequences": ["User:"],
            "temperature": 0,
            "topP": 1
            
        }
        
    }

    # convert payload to json
    body = json.dumps(payload)

    # set model id we want to use from bedrock
    model_id = "amazon.titan-text-lite-v1"

    # invoke the model to get a response
    response = bedrock.invoke_model(
           body=body,
           modelId=model_id,
           accept="application/json",
           contentType="application/json"
    )

    # get response from returned json
    response_body = json.loads(response.get("body").read())
    response_text = response_body['results'][0]['outputText']

    # add metadata to response for reference
    response_text = f"{response_text}, metadata: {"".join(str(metadata))}"
    
    return response_text

question = st.text_input(label="input what type of toy you are looking for or what information you are looking to learn about a toy", 
                             max_chars=100, key='question', type='default')
    
if st.button("Submit Question"):
    data_load_state = st.text('Getting response...')
    results = get_products_information(question, embeddings, vector_store, bedrock)
    data_load_state.text("Done getting response")
    st.write(results)
