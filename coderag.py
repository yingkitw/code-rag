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
import time
import shutil
import tempfile

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

def read_code_in_folder(folderpath, chunk_size=250, chunk_overlap=20, summary_dict=None):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = []

    for root, _, files in os.walk(folderpath):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            base_name = os.path.basename(file_path)
            if base_name not in summary_dict:
                continue

            encoding = detect_encoding(file_path)
            try:
                # Count lines in the file
                with open(file_path, 'r', encoding=encoding) as f:
                    line_count = sum(1 for _ in f)
                
                text = TextLoader(file_path, encoding=encoding).load()
                chunks = text_splitter.split_documents(text)
                docs += chunks
                summary_dict[base_name].update({
                    'Lines': line_count,
                    'Status': '✅ Processed'
                })
            except Exception as e:
                summary_dict[base_name].update({
                    'Lines': 0,
                    'Status': f'❌ Failed: {str(e)}'
                })

    URI = "./code_rag.db"
    vector_store = Milvus(
        embedding,
        connection_args={"uri": URI},
        collection_name="code",
        auto_id=True,
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
        auto_id=True,
    )
    docs = vector_store.similarity_search(query)
    answer = querycode(docs, query)

    return answer

def read_file_in_encoding(file_path):
    encoding = detect_encoding(file_path)
    with open(file_path, 'r',encoding=encoding) as f:
        return f.read()

def convert_code(file_path, target_language):
    # st.info(f"Converting {file_name}...")
    prompt = f"""<|system|>
you are expert on programming
please convert the code provided in the file to target language.
<<SYS>>
source code:`{read_file_in_encoding(file_path)}`
target language:`{target_language}`
<<SYS>>
<|assistant|>
answer:
"""
    answer = wx.watsonx_gen_stream(prompt, wx.GRANITE_3_8B_INSTRUCT)
    return answer

def build_code(file_name):
    st.info(f"Building {file_name}...")
    # Placeholder for code building functionality
    pass

def test_code(file_name):
    st.info(f"Testing {file_name}...")
    # Placeholder for code testing functionality
    pass

def revise_code(file_name):
    st.info(f"Revising {file_name}...")
    # Placeholder for code revision functionality
    pass

def process_uploaded_files(uploaded_files):
    if uploaded_files:
        all_files = []
        summary_dict = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_file in uploaded_files:
                file_name = os.path.join(temp_dir, uploaded_file.name)
                try:
                    # Save the uploaded file locally
                    with open(file_name, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process ZIP files
                    if zipfile.is_zipfile(file_name):
                        with zipfile.ZipFile(file_name, 'r') as zip_ref:
                            extract_folder = os.path.join(temp_dir, f"extracted_{uploaded_file.name.split('.')[0]}")
                            zip_ref.extractall(extract_folder)
                            
                            # Add extracted files
                            for root, _, files in os.walk(extract_folder):
                                for extracted_file in files:
                                    if is_source_file(extracted_file):
                                        extracted_path = os.path.join(root, extracted_file)
                                        all_files.append(extracted_path)
                                        summary_dict[extracted_file] = {
                                            'File': extracted_file,
                                            'Source': uploaded_file.name,
                                            'Type': os.path.splitext(extracted_file)[1].lower(),
                                            'Lines': 0,
                                            'Status': '📦 Extracted'
                                        }
                                    else:
                                        summary_dict[extracted_file] = {
                                            'File': extracted_file,
                                            'Source': uploaded_file.name,
                                            'Type': os.path.splitext(extracted_file)[1].lower() or 'No ext',
                                            'Lines': 0,
                                            'Status': '⚠️ Unsupported'
                                        }
                    else:
                        if is_source_file(file_name):
                            all_files.append(file_name)
                            summary_dict[uploaded_file.name] = {
                                'File': uploaded_file.name,
                                'Source': 'Direct',
                                'Type': os.path.splitext(file_name)[1].lower(),
                                'Lines': 0,
                                'Status': '📄 Uploaded'
                            }
                        else:
                            summary_dict[uploaded_file.name] = {
                                'File': uploaded_file.name,
                                'Source': 'Direct',
                                'Type': os.path.splitext(file_name)[1].lower() or 'No ext',
                                'Lines': 0,
                                'Status': '⚠️ Unsupported'
                            }
                except Exception as e:
                    summary_dict[uploaded_file.name] = {
                        'File': uploaded_file.name,
                        'Source': 'Failed',
                        'Type': os.path.splitext(file_name)[1].lower() if os.path.splitext(file_name)[1] else 'No ext',
                        'Lines': 0,
                        'Status': '❌ Error'
                    }

            # Process valid files and update summary
            if all_files:
                with st.spinner("⏳ Processing files..."):
                    read_code_in_folder(temp_dir, summary_dict=summary_dict)

            # Display unified summary table with action buttons
            if summary_dict:
                st.write("### 📊 File Processing Summary")
                
                # Add action buttons and language selector at the top
                st.write("### Actions")
                
                # Add target language selector
                target_language = st.selectbox(
                    "Select target language for conversion",
                    ["Python", "Java", "C++", "JavaScript", "TypeScript", "Go", "Rust", "C#"]
                )
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button("Convert Selected"):
                        # Add user action to chat history
                        if "messages" not in st.session_state:
                            st.session_state.messages = []
                        
                        selected_files = [file_name for file_name, file_info in summary_dict.items() 
                                       if st.session_state.get(f"select_{file_name}", False)]
                        
                        if selected_files:
                            user_message = f"Convert the following files to {target_language}: {', '.join(selected_files)}"
                            st.session_state.messages.append({"role": "user", "content": user_message})

                            for file_name, file_info in summary_dict.items():
                                if st.session_state.get(f"select_{file_name}", False):
                                    # Create a new expander for each file's conversion result
                                    file_path = os.path.join(temp_dir, file_name)
                                    response = ""
                                    for part in convert_code(file_path, target_language):
                                        response += part
                                    st.session_state.messages.append({"role": "bot", "content": response})
                            
                
                with col2:
                    if st.button("Build Selected"):
                        for file_name, file_info in summary_dict.items():
                            if st.session_state.get(f"select_{file_name}", False):
                                build_code(file_name)
                
                with col3:
                    if st.button("Test Selected"):
                        for file_name, file_info in summary_dict.items():
                            if st.session_state.get(f"select_{file_name}", False):
                                test_code(file_name)
                
                with col4:
                    if st.button("Revise Selected"):
                        for file_name, file_info in summary_dict.items():
                            if st.session_state.get(f"select_{file_name}", False):
                                revise_code(file_name)
                
                st.markdown("---")
                
                # Create columns for the table
                col1, col2, col3, col4, col5, col6 = st.columns([1, 2, 1, 1, 1, 1])
                
                # Headers
                col1.write("**Select**")
                col2.write("**File**")
                col3.write("**Source**")
                col4.write("**Type**")
                col5.write("**Lines**")
                col6.write("**Status**")
                
                # Display each file with checkboxes
                for file_name, file_info in summary_dict.items():
                    # Initialize checkbox state in session state if not exists
                    if f"select_{file_name}" not in st.session_state:
                        st.session_state[f"select_{file_name}"] = False
                    
                    # Only show checkboxes for supported files
                    if is_source_file(file_info['File']):
                        col1.checkbox("", key=f"select_{file_name}")
                    else:
                        col1.write("")
                    
                    col2.write(file_info['File'])
                    col3.write(file_info['Source'])
                    col4.write(file_info['Type'])
                    col5.write(str(file_info['Lines']))
                    col6.write(file_info['Status'])
                
                total_lines = sum(item['Lines'] for item in summary_dict.values())
                st.write(f"Total lines processed: {total_lines}")
            else:
                st.warning("No valid files to process. Please upload C, Java, COBOL, or C++ files.")
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

        with st.chat_message("bot"):
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
        type=["c","h","java","cbl","cpp","hpp","zip"],
        accept_multiple_files=True,
        help="Only C, Java, COBOL, and header files are supported"
    )

    process_uploaded_files(uploaded_files)
    chat_interface()

if __name__ == "__main__":
    main()