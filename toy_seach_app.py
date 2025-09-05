import streamlit as st
import pandas as pd 
import numpy as np
import boto3
import yaml
import json

from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

st.title('Toy Shopping App')

st.write("""This app was created as an example of how to use a ratrieval augmented generation (RAG) model to help customers answer questions related to a stores toy inventory. This app uses the 
            retail toy dataset which can be found here https://github.com/GoogleCloudPlatform/python-docs-samples/blob/main/cloud-sql/postgres/pgvector/data/retail_toy_dataset.csv . 
            The product description has been encoded into sentence embeddings using the OpenAIEMbeddings model and stored in a pinecone vector database. The user can ask any question
            about the toy dataset and the model will find the most relevant chunks and inject this information into a prompt to the 2.7B parameter from meta 
            (found here https://huggingface.co/meta-llama/Llama-2-7b) to provide an answer based on product descriptions price and model number.""")

st.write("The code to create this app can be found here https://github.com/kyle-becker/llm_rag_example")


# Download and save the dataset containing product information in a Pandas dataframe.
DATASET_URL='https://github.com/GoogleCloudPlatform/python-docs-samples/raw/main/cloud-sql/postgres/pgvector/data/retail_toy_dataset.csv'
df = pd.read_csv(DATASET_URL)

#filter for specific columns
df = df.loc[:, ['product_id', 'product_name', 'description', 'list_price']]

st.write(df.head(5))

st.write(""" An example of a prompt you can use is "I am looking for Bicycle Playing cards, what is the product_name and list_price" """)

# connect to pinecone
pc = Pinecone(api_key=st.secrets['pinecone_api_key'])

#connect to openai for document embeddings
embeddings = OpenAIEmbeddings(api_key=st.secrets['open_ai_api_key'])

# set up vector store to upsert to index and then embedd the documents using openAI Embedding from above
index = pc.Index('product-descriptions')
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# connect to  amazon bedrock
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1", aws_access_key_id=st.secrets['aws_access_key_id'],
        aws_secret_access_key=st.secrets['aws_secret_access_key'])

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

question = st.text_input(label="Tell me what type of toy you are looking for and any information you would like to know about it", 
                             max_chars=200, key='question', type='default')
    
if st.button("Submit Question"):
    data_load_state = st.text('Getting response...')
    results = get_products_information(question, embeddings, vector_store, bedrock)
    data_load_state.text("Done getting response")
    st.write(results)
