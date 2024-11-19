import os
from git import Repo
from urllib.parse import urlparse
import streamlit as st
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
import vertexai
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Initialize Vertex AI settings
PROJECT_ID = "coderefactoringai"
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Streamlit app title
st.title("GitHub Repo Q&A App")

# Function to generate repo path
def generate_repo_path(url):
    repo_name = urlparse(url).path.split('/')[-1].replace(".git", "")
    return os.path.join("/Users/soemoe/Downloads/Github Repo", repo_name)

# Input field for the repository URL
repo_url = st.text_input("Enter the GitHub repository URL (Python projects only):")


# Button to initiate the process
if st.button("Load and Process Repository"):
    if repo_url:
        try:
            # Generate a unique path for the repository
            repo_path = generate_repo_path(repo_url)
            st.write(f"Cloning repository into: {repo_path}")

            # Clone the repository
            repo = Repo.clone_from(repo_url, to_path=repo_path)
            st.write("Repository cloned successfully!")

            # Load and parse documents
            loader = GenericLoader.from_filesystem(
                repo_path,
                glob="**/*",
                suffixes=[".py"],
                exclude=["**/non-utf8-encoding.py"],
                parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
            )
            documents = loader.load()
            st.write(f"Number of Parsed Documents: {len(documents)}")

            # Split documents into manageable chunks
            python_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
            )
            texts = python_splitter.split_documents(documents)
            st.write(f"Number of Split Texts: {len(texts)}")

            # Initialize embeddings
            embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

            # Create Chroma database from documents
            db = Chroma.from_documents(
                texts,
                embeddings,
                persist_directory=os.path.join(os.getcwd(), "chroma_db")  # Path relative to current working directory
            )

            retriever = db.as_retriever(
                search_type="mmr", search_kwargs={"k": 8}
            )

            # Set up the LLM with Vertex AI
            llm = VertexAI(
                project=PROJECT_ID,
                location=LOCATION,
                model="gemini-1.5-flash-002",
                model_kwargs={
                    "temperature": 0.7,
                    "max_length": 600,
                    "top_p": 0.95,
                    "top_k": 50
                }
            )

            # Define prompts
            search_prompt = ChatPromptTemplate.from_messages([
                ("placeholder", "{chat_history}"),
                ("user", "{input}"),
                (
                    "user",
                    "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
                ),
            ])
            retriever_chain = create_history_aware_retriever(llm, retriever, search_prompt)

            answer_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "Answer the user's questions based on the below context:\n\n{context}",
                ),
                ("placeholder", "{chat_history}"),
                ("user", "{input}"),
            ])
            document_chain = create_stuff_documents_chain(llm, answer_prompt)
            qa = create_retrieval_chain(retriever_chain, document_chain)

            st.success("Setup complete. You can now ask questions!")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Only show the question input and "Get Answer" button if qa is initialized
if qa:
    question = st.text_input("Ask a question about the repository (or type 'exit' to quit):")

    if st.button("Get Answer"):
        if question and question.lower() != 'exit':
            try:
                result = qa.invoke({"input": question})
                if result:
                    st.write("Answer:", result)
                else:
                    st.write("Sorry, I couldn't find an answer.")
            except Exception as e:
                st.error(f"An error occurred while retrieving the answer: {str(e)}")
        elif question.lower() == 'exit':
            st.write("Goodbye!")
else:
    st.warning("Please load and process the repository first to ask questions.")
