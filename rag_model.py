from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
import streamlit as st
import os


#----------------- Setting -------------------------
api_key_openai = st.secrets["api_key_openai"]
api_key_pinecone = st.secrets["api_key_pinecone"]
directory = st.secrets["directory"]
index_name=st.secrets["index_name"]
#----------------- Setting -------------------------
 
def setup_openai_api(api_key):
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = api_key_openai
        print("OPENAI_API_KEY has been set!")
    else:
        exit
    
 
def setup_pinecone_api(api_key):
    if "PINECONE_API_KEY" not in os.environ:
        os.environ["PINECONE_API_KEY"] = api_key_pinecone
    else:
        exit

    
 
def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents
 
def chunk_data(documents, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    return chunks
 
def initialize_embeddings():
    embeddings = OpenAIEmbeddings()
    return embeddings
 

# if "context_mem" not in st.session_state:
#     memory= ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)
#     st.session_state["context_mem"] = memory
# else:
#     memory=st.session_state["context_mem"]



def create_retrieval_chain(vectorstore, embeddings, memory):
    chain = ConversationalRetrievalChain.from_llm(
        OpenAI(), 
        vectorstore.as_retriever(search_kwargs={'k':3}),
        memory=memory,
        condense_question_prompt=condense_question_prompt_template,
        combine_docs_chain_kwargs=dict(prompt=qa_prompt),
        verbose=True
    )
    return chain
 

chat_history=[]

def perform_conversational_retrieval(chain, query):
    if "context_mem" not in st.session_state:
        memory= ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)
        st.session_state["context_mem"] = memory
    else:
        memory=st.session_state["context_mem"]
    output = chain(query, memory)
    chat_history.append(output)
    return output



setup_openai_api(api_key_openai)
setup_pinecone_api(api_key_pinecone)
documents = read_doc(directory)
chunks = chunk_data(documents)
embeddings = initialize_embeddings()




def ask_model(api_key_openai, api_key_pinecone, directory):
    
    #index = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
    vectorstore = Pinecone.from_existing_index(index_name=index_name,embedding = embeddings, namespace="")
    if "context_mem" not in st.session_state:
        memory= ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)
        st.session_state["context_mem"] = memory
    else:
        memory=st.session_state["context_mem"]
    chain = create_retrieval_chain(vectorstore,embeddings, memory=memory)
    return chain



_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a 
standalone question without changing the content in given question. If the question is not related to previous context, Please DO NOT rephrase it. Answer the question directly.
examples = [
    [
        "question":explain Economy?,
        "answer":
Follow up:explain it in terms  of green growth? 
standalone question:explain economy in terms  of green growth?
Follow up:under which scheme we provide food to 80 crore people in pandamic?
standalone question:under which scheme we provide food to 80 crore people in pandamic?
    ],
    [
        "question":under which scheme we provide food to 80 crore people in pandamic?
        "answer":
Follow up:what is the entire expenditure  in that scheme?
standalone question:what is the entire expenditure in PMGKAY scheme?
Follow up:Define PM Awas Yojana?
standalone question:Define PM Awas Yojana?
    ],
    [
        "question":what are the types of leaves?
        "answer":
Follow up: tell me about maternity leave?
standalone question:explain in brief about maternity leave?
    ]

]   

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

condense_question_prompt_template = PromptTemplate.from_template(_template)

# """You are helpful information giving QA System for HR Policies. Make sure you don't answer anything not related \
# to following context. You will have useful information & details available in the given context. Use the following\
# piece of context to answer the question at the end. If you don't know the answer, just say that you don't know. Don't\
# try to make up an answer. If the question is not related to previous context,Please DO NOT rephrase it. Answer the question directly."""



prompt_template = """You are a helpful information giving QA assistant for HR Policies. You will be given the Policies in the context below.\
You know about the leaves, attendance, loan and every other policy given in the context. 
Make sure you don't answer anything not related to following context. If you don't know the answer,\
just say that you don't know. Don't try to make up an answer. If the question is not related to\
previous context,Please DO NOT rephrase it. Answer the question directly.

{context}

Question: {question}
Helpful Answer:"""


qa_prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
 