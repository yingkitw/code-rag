import streamlit as st
from dotenv import load_dotenv
from watsonx import WatsonxAI
import warnings
import os
import chardet
import zipfile
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus

warnings.filterwarnings("ignore")

proxy = "proxy.us.ibm.com:8080"

wx = WatsonxAI()
wx.connect()

load_dotenv()

api_key = os.getenv("API_KEY", None)
project_id = os.getenv("PROJECT_ID", None)

creds = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": api_key
}

embed_params = {
    EmbedParams.TRUNCATE_INPUT_TOKENS: 3,
    EmbedParams.RETURN_OPTIONS: {
        'input_text': True
    }
}

embedding = Embeddings(
    model_id="sentence-transformers/all-minilm-l12-v2",
    params=embed_params,
    credentials=creds,
    project_id=project_id
)

conversation_history = []

def detect_encoding(filename):
    detected_encoding = ""
    with open(filename, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        detected_encoding = result.get("encoding")
    return detected_encoding

def read_code_in_folder(folderpath, chunk_size=250, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = []

    for root, _, files in os.walk(folderpath):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            encoding = detect_encoding(file_path)
            text = TextLoader(file_path, encoding=encoding).load()
            docs += text_splitter.split_documents(text)

    URI = "./code_rag.db"
    vector_store = Milvus(
        embedding,
        connection_args={"uri": URI},
        collection_name="code",
    )
    vector_store.add_documents(documents=docs)
    print(st.session_state.db)

def querycode(informations, question):
    prompt = f"""<|system|>
you are expert on programming
-make your best guess on what programming language in the code provided.
<|user|>
please answer the question in the programming logic for the source code in backquoted.
<<SYS>>
source code:`{informations}`
question:`{question}`
<<SYS>>
<|assistant|>
answer:
"""
    answer = wx.watsonx_gen_stream(prompt, wx.GRANITE_3_8B_INSTRUCT)
    return answer

def check_code_quality(file_path):
    return "it is a source file"  # Placeholder for demonstration

def gen_answer(query):
    URI = "./code_rag.db"

    vector_store = Milvus(
        embedding,
        connection_args={"uri": URI},
        collection_name="code",
    )
    docs = vector_store.similarity_search(query)
    answer = querycode(docs, query)

    return answer

def chat_interface():
    st.title("Chat with Code")

    with st.chat_message("bot"):
        st.write("Hello 👋, it is a chatbot base on the code you uploaded.")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What may I help you"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role":"user", "content":prompt})

        st.session_state.prompt = prompt

        with st.chat_message("assistant"):
            response = st.write_stream(gen_answer(prompt))
        # response = gen_answer(prompt)
        st.session_state.messages.append({"role":"bot", "content":response})

def process_uploaded_files(uploaded_files):
    if uploaded_files:
        all_files = []

        for uploaded_file in uploaded_files:
            file_name = "sample/"+uploaded_file.name
            st.write(f"Processing file: {file_name}")

            # Save the uploaded file locally
            with open(file_name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.write(f"Saved file: {file_name}")

            # If the file is a ZIP, extract it
            if zipfile.is_zipfile(file_name):
                st.write(f"Extracting ZIP file: {file_name}")
                with zipfile.ZipFile(file_name, 'r') as zip_ref:
                    extract_folder = f"extracted_{file_name.split('.')[0]}"
                    zip_ref.extractall(extract_folder)
                    st.write(f"Extracted files to folder: {extract_folder}")

                    # Add extracted files to the list
                    for root, dirs, files in os.walk(extract_folder):
                        for extracted_file in files:
                            extracted_file_path = os.path.join(root, extracted_file)
                            all_files.append(extracted_file_path)
                            st.write(f"Extracted file: {extracted_file_path}")
            else:
                # If it's not a ZIP, add directly to the list
                all_files.append(file_name)

        # Display all processed files
        st.write(f"Total files processed: {len(all_files)}")
        for file in all_files:
            st.write(f"- {file}")

        # Example: Add custom logic to process the files
        # for file in all_files:
        #     status = check_code_quality(file)
        #     st.write(f"File: {file} - Status: {status}")
    else:
        st.write("No files uploaded. Please upload files or a zipped folder.")

def main():
    st.sidebar.title("Upload Files")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Files or Zipped Folders",
        type=None,  # Allow any file type
        accept_multiple_files=True  # Enable multiple file uploads
    )

    process_uploaded_files(uploaded_files)

    chat_interface()

if __name__ == "__main__":
    main()