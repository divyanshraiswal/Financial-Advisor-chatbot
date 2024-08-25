import os
import boto3
import streamlit as st

## We will be suing Titan Embeddings Model To generate Embedding

from langchain_aws import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock
from langchain_aws import BedrockLLM

## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding And Vector Store

from langchain_community.vectorstores import FAISS

## LLm Models

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime",region_name="us-east-1")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)


## Data ingestion
def data_ingestion():
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()

    # - in our testing Character split works better with this PDF data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    docs=text_splitter.split_documents(documents)
    return docs

## Vector Embedding and vector store

def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")


def get_MistralAI_llm():
    #load the mistral express model
    llm=BedrockLLM(model_id="mistral.mistral-large-2402-v1:0",client=bedrock,
                model_kwargs={'max_tokens':200})
    
    return llm

def get_llama3_llm():
    ##create the llama3 Model
    llm=BedrockLLM(model_id="meta.llama3-8b-instruct-v1:0",client=bedrock,
                model_kwargs={'max_gen_len':512})
    
    return llm

prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.

<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa.invoke({"query":query})
    return answer['result']


os.environ['PYTHONWARNINGS'] = 'ignore:Pickle deserialization is dangerous'

def load_faiss_index(index_path):
    try:
        faiss_index = FAISS.load_local(index_path, bedrock_embeddings,allow_dangerous_deserialization=True)
        return faiss_index
    except ValueError as e:
        st.error(f"Error loading FAISS index: {e}")
        return None

def main():
    st.set_page_config("Chat PDF")

    st.header("Chat with Financial Chatbot to answer your questions")

    user_question = st.text_input("Ask a Question from the PDF Files ")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Llama3 Output"):
        with st.spinner("Processing..."):
            faiss_index = load_faiss_index("faiss_index")
            if faiss_index is not None:
                llm = get_llama3_llm()
                st.write(get_response_llm(llm, faiss_index, user_question))
                st.success("Done")

    if st.button("Mistral AI output"):
        with st.spinner("Processing..."):
            faiss_index = load_faiss_index("faiss_index")
            if faiss_index is not None:
                llm = get_MistralAI_llm()
                st.write(get_response_llm(llm, faiss_index, user_question))
                st.success("Done")


if __name__ == "__main__":
    main()
