This app was created as an example of how to use a ratrieval augmented generation (RAG) model to help customers answer questions related to a stores toy inventory. This app uses the 
retail toy dataset which can be found here https://github.com/GoogleCloudPlatform/python-docs-samples/blob/main/cloud-sql/postgres/pgvector/data/retail_toy_dataset.csv . 
The product description has been encoded into sentence embeddings using the OpenAIEMbeddings model and stored in a pinecone vector database. The user can ask any question
about the toy dataset and the model will find the most relevant chunks and inject this information into a prompt to the 2.7B parameter from meta 
(found here https://huggingface.co/meta-llama/Llama-2-7b) to provide an answer based on product descriptions price and model number.

An example of a prompt you can use is "I am looking for Bicycle Playing cards, what is the product_name and list_price"