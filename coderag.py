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

def is_source_file(filename):
    """Check if a file is a source code file based on its extension."""
    # First check if file has an extension
    if not os.path.splitext(filename)[1]:
        return False
        
    source_extensions = {
        '.c',     # C
        '.h',     # Header
        '.java',  # Java
        '.cbl',   # COBOL
        '.cob',   # COBOL
        '.cpp',   # C++
        '.hpp'    # C++ header
    }
    return os.path.splitext(filename)[1].lower() in source_extensions

def read_code_in_folder(folderpath, chunk_size=250, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = []
    process_summary = []

    for root, _, files in os.walk(folderpath):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            encoding = detect_encoding(file_path)
            try:
                text = TextLoader(file_path, encoding=encoding).load()
                chunks = text_splitter.split_documents(text)
                docs += chunks
                process_summary.append({
                    'File': os.path.basename(file_path),
                    'Type': os.path.splitext(file_path)[1].lower(),
                    'Chunks': len(chunks),
                    'Status': '✅ Processed'
                })
            except Exception as e:
                process_summary.append({
                    'File': os.path.basename(file_path),
                    'Type': os.path.splitext(file_path)[1].lower(),
                    'Chunks': 0,
                    'Status': f'❌ Failed: {str(e)}'
                })

    # Display processing summary in a table
    if process_summary:
        st.write("### 📊 Processing Summary")
        st.table(process_summary)
        total_chunks = sum(item['Chunks'] for item in process_summary)
        st.write(f"Total chunks indexed: {total_chunks}")

    URI = "./code_rag.db"
    vector_store = Milvus(
        embedding,
        connection_args={"uri": URI},
        collection_name="code",
        auto_id=True,
        drop_old=True
    )
    vector_store.add_documents(documents=docs)

def querycode(informations, question):
    prompt = f"""<|system|>
you are expert on programming
-make your best guess on what programming language in the code provided.
 
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

def process_uploaded_files(uploaded_files):
    if uploaded_files:
        all_files = []
        upload_summary = []

        os.makedirs("sample", exist_ok=True)
        
        # Create a DataFrame-like structure for the summary
        for uploaded_file in uploaded_files:
            file_name = "sample/" + uploaded_file.name
            try:
                # Save the uploaded file locally
                with open(file_name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process ZIP files
                if zipfile.is_zipfile(file_name):
                    with zipfile.ZipFile(file_name, 'r') as zip_ref:
                        extract_folder = f"extracted_{file_name.split('.')[0]}"
                        zip_ref.extractall(extract_folder)
                        
                        # Add extracted files
                        for root, _, files in os.walk(extract_folder):
                            for extracted_file in files:
                                if is_source_file(extracted_file):
                                    extracted_path = os.path.join(root, extracted_file)
                                    all_files.append(extracted_path)
                                    upload_summary.append({
                                        'File': os.path.basename(extracted_file),
                                        'Source': uploaded_file.name,
                                        'Type': os.path.splitext(extracted_file)[1].lower(),
                                        'Status': '📦 Extracted'
                                    })
                                else:
                                    upload_summary.append({
                                        'File': os.path.basename(extracted_file),
                                        'Source': uploaded_file.name,
                                        'Type': os.path.splitext(extracted_file)[1].lower() or 'No ext',
                                        'Status': '⚠️ Unsupported'
                                    })
                else:
                    if is_source_file(file_name):
                        all_files.append(file_name)
                        upload_summary.append({
                            'File': uploaded_file.name,
                            'Source': 'Direct',
                            'Type': os.path.splitext(file_name)[1].lower(),
                            'Status': '📄 Uploaded'
                        })
                    else:
                        upload_summary.append({
                            'File': uploaded_file.name,
                            'Source': 'Direct',
                            'Type': os.path.splitext(file_name)[1].lower() or 'No ext',
                            'Status': '⚠️ Unsupported'
                        })
            except Exception as e:
                upload_summary.append({
                    'File': uploaded_file.name,
                    'Source': 'Failed',
                    'Type': os.path.splitext(file_name)[1].lower() if os.path.splitext(file_name)[1] else 'No ext',
                    'Status': '❌ Error'
                })

        # Display upload summary in a table
        if upload_summary:
            st.write("### 📤 Upload Summary")
            st.table(upload_summary)
            
            # Process valid files
            if all_files:
                with st.spinner("⏳ Processing files..."):
                    read_code_in_folder("sample")
            else:
                st.warning("No valid files to process. Please upload C, Java, COBOL, or header files.")
    else:
        st.info("ℹ️ Upload files to begin", icon="ℹ️")

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

def main():
    st.set_page_config(page_title="Code RAG", layout="wide")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    st.sidebar.title("Upload Files")
    st.sidebar.markdown("""
    Supported: `.c` `.h` `.java` `.cbl` `.cpp` `.hpp`
    ZIP files accepted.
    """)
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload Files or Folders",
        type=["c","h","java","cbl","cpp","hpp","zip"],  # We'll handle file type validation in the code
        accept_multiple_files=True,
        help="Only C, Java, COBOL, and header files are supported"
    )

    process_uploaded_files(uploaded_files)
    chat_interface()

if __name__ == "__main__":
    main()