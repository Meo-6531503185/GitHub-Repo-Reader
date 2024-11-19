import os
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
import vertexai
import google.auth 
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings
from urllib.parse import urlparse 
import streamlit as st

def generate_repo_path(url):
    # Extract the repository name from the URL
    repo_name = urlparse(url).path.split('/')[-1].replace(".git", "")
    return os.path.join("/Users/soemoe/Downloads/Github Repo", repo_name)

# Taking the repository URL as input from the user
repo_url = input("Please enter the repository URL <<Python Project Only>> : ")

# Generate a unique path for the repository
repo_path = generate_repo_path(repo_url)

# Clone the repository into the unique path
repo = Repo.clone_from(repo_url, to_path=repo_path)

loader = GenericLoader.from_filesystem(
    repo_path,
    glob="**/*",
    suffixes=[".py"],
    exclude=["**/non-utf8-encoding.py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
)
documents = loader.load()
print("Number of Parsed-Documents = ",len(documents))

from langchain_text_splitters import RecursiveCharacterTextSplitter

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
texts = python_splitter.split_documents(documents)
print("Number of Splitted-Text from Parsed-Documents = ",len(texts))

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
PROJECT_ID = "coderefactoringai"  
LOCATION = "us-central1"  
vertexai.init(project=PROJECT_ID, location=LOCATION)
embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

db = Chroma.from_documents(texts, embeddings)
retriever = db.as_retriever(
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 8},
)

from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = VertexAI(
        project="coderefactoringai",
        location="us-central1",  # Common location for Vertex AI
        model="gemini-1.5-flash-002",  # Specify the Gemini model
        model_kwargs={
            "temperature": 0.7,
            "max_length": 600,
            "top_p": 0.95,
            "top_k": 50
        }
)

# First we need a prompt that we can pass into an LLM to generate this search query

prompt = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
        (
            "user",
            "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
        ),
    ]
)

retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's questions based on the below context:\n\n{context}",
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ]
)
document_chain = create_stuff_documents_chain(llm, prompt)

qa = create_retrieval_chain(retriever_chain, document_chain)


while True:
    question = st.input("Please ask a question (or type 'exit' to quit): ")
    
    if question.lower() == 'exit':
        print("Goodbye!")
        break  # Exit the loop if the user types 'exit'

    result = qa.invoke({"input": question})
    answer = result.get('answer')
    
    if answer:
        print("Answer:", answer)
    else:
        print("Sorry, I couldn't find an answer.")

