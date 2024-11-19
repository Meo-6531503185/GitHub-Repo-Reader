import os
import streamlit as st
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
import vertexai
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.auth 
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_chroma import Chroma
from urllib.parse import urlparse 
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from htmlTemplates import css,bot_template,user_template
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


def generate_repo_path(url):
    # Extract the repository name from the URL
    repo_name = urlparse(url).path.split('/')[-1].replace(".git", "")
    return os.path.join("/Users/soemoe/Downloads/Github Repo", repo_name)

def get_parsedDocuments(repo_url):
    
    repo_path = generate_repo_path(repo_url)
    repo = Repo.clone_from(repo_url, to_path=repo_path)

    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".py"],
        exclude=["**/non-utf8-encoding.py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
    )
    documents = loader.load()
    return documents

def get_splittedContents(documents):

    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
    )
    contents = python_splitter.split_documents(documents)
    return contents


def get_vectorStores(texts):

    PROJECT_ID = "coderefactoringai"  
    LOCATION = "us-central1"  
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    embedding = VertexAIEmbeddings(model_name="text-embedding-004")
    vectorStore = Chroma.from_documents(texts, embedding)
    return vectorStore

def handle_userinput(user_question):
    response = st.session_state.conversation.invoke({'question': user_question})
    #st.write(response)
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def get_conversation_chain(vector_store):
    retriever = vector_store.as_retriever(
        search_type="mmr",  
        search_kwargs={"k": 8},
    )
    llm = VertexAI(
            project="coderefactoringai",
            location="us-central1",  
            model="gemini-1.5-flash-002",  
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
    return qa


        

def main():

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    st.header("Github Repository Reader.")  
    user_question = st.text_input("Ask a question about the repository:")
    if user_question:
        handle_userinput(user_question)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    

    with st.sidebar:
        st.subheader("1. Upload a URL of the Github repository you want to read")
        repo_url = st.text_input("Enter the GitHub repository URL:")
        if st.button("Upload"):
            with st.spinner("Reading"):
                parsed_documents = get_parsedDocuments(repo_url)
                
                splitted_contents = get_splittedContents(parsed_documents)
                
                vector_store = get_vectorStores(splitted_contents)
                
                if vector_store is None:
                    st.error("Failed to create embeddings. Please check your input data.")
                else:
                    st.session_state.conversation = get_conversation_chain(vector_store)


if __name__ == "__main__":  
    main()

"""while True:
        question = input("Please ask a question (or type 'exit' to quit): ")
        
        if question.lower() == 'exit':
            print("Goodbye!")
            break  # Exit the loop if the user types 'exit'

        result = qa.invoke({"input": question})
        answer = result.get('answer')
        
        if answer:
            print("Answer:", answer)
        else:
            print("Sorry, I couldn't find an answer.")"""