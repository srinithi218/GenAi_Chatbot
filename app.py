import os
import streamlit as st
from supabase import create_client

# Modern Google & Classic LangChain integrations
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Classic compatibility packages for legacy chain architectures
from langchain_classic.chains import RetrievalQA

# 🎬 1. App Layout and Configuration
st.set_page_config(page_title="Dynamic PDF AI Assistant", page_icon="🤖", layout="wide")
st.title("📚 Custom Knowledge Base Chatbot")
st.write("Upload a PDF document, build the index, and chat with your data!")

# Initialize Persistent Session Chat History Memory
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# 🎬 2. Safe Secrets Verification
GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

# Guarantee environment bindings for native LangChain dependencies
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Persistent Directory Setup for Local Index Caching
FAISS_INDEX_PATH = "faiss_indexes/my_subject_index"
os.makedirs("temp_docs", exist_ok=True)


# 🎬 3. Core RAG Compilation Engine
def process_and_index_pdf(uploaded_file):
    """Saves the uploaded buffer, splits chunks, and builds the dual retrieval layers."""
    temp_path = os.path.join("temp_docs", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("⚡ Parsing textual elements via PyMuPDF...")
    loader = PyMuPDFLoader(temp_path)
    docs = loader.load_and_split()

    if not docs or all(not doc.page_content.strip() for doc in docs):
        st.error("❌ The PDF file appears to be completely empty or an unreadable image format.")
        return None

    # Text Chunk Splitting
    text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
    split_chunks = text_splitter.split_documents(docs)
    st.info(f"🧱 Partitioned script into {len(split_chunks)} functional content chunks.")

    # Dense Vector Setup (Gemini Embeddings)
    st.info("🚀 Generating vector embeddings with gemini-embedding-2...")
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=GOOGLE_API_KEY
    )
    vector_db = FAISS.from_documents(split_chunks, embedding_function)
    vector_db.save_local(FAISS_INDEX_PATH)
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # Initialize Gemini 2.5 Flash Model Core
    # llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=GOOGLE_API_KEY)
    # st.info("🔥 Powering up Llama 3.3 70B on Groq LPUs...")
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        groq_api_key=GROQ_API_KEY,
        temperature=0.7
    )

    # Build prompt configurations
    template = """You are a question bank expert chatbot. Use only the source data provided to answer the queries.
If the answer to user query is not in the source data or is incomplete, say:
"I’m sorry, but I couldn’t find the information in the provided data."

{context}

Question: {question}
Answer:"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Construct complete QA Engine
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_retriever,
        chain_type_kwargs={"prompt": prompt, "document_variable_name": "context"}
    )
    return chain


# 🎬 4. Frontend Sidebar Layout Control
with st.sidebar:
    st.header("🗂️ Knowledge Provisioning")
    uploaded_file = st.file_uploader("Upload reference manual (PDF)", type=["pdf"])

    if uploaded_file:
        if st.button("🏗️ Build Knowledge Base"):
            with st.spinner("Processing document layout..."):
                compiled_chain = process_and_index_pdf(uploaded_file)
                if compiled_chain:
                    st.session_state["rag_chain"] = compiled_chain
                    # Clear past records if a brand new knowledge layout script is loaded
                    st.session_state["chat_history"] = [] 
                    st.success("🎉 Engine operational! Ready to handle prompts.")

# 🎬 5. Main Screen Execution & Continuous Scrolling Chat UI
if "rag_chain" in st.session_state:
    
    # Render the continuous historical thread timeline first
    # This automatically stacks your chat bubbles chronologically down the page
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Use the dedicated chat input bar placed naturally at the bottom of the viewport
    if user_query := st.chat_input("Ask something about the uploaded document..."):
        
        # 1. Instantly render and log user prompt to the browser interface layout
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state["chat_history"].append({"role": "user", "content": user_query})

        # 2. Spin up inference layers to append the script reply block
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("🤔 Consultation in progress..."):
                try:
                    # Trigger structural dynamic pipeline execution
                    chain_response = st.session_state["rag_chain"].invoke({"query": user_query})
                    response_text = chain_response["result"]

                    # Flash the text onto the browser view interface smoothly
                    response_placeholder.markdown(response_text)
                    
                    # Store to history stack frame to maintain scrolling visibility on next rerun
                    st.session_state["chat_history"].append({"role": "assistant", "content": response_text})

                    # 3. Synchronously arch new metadata context down into your Supabase log engine
                    log_data = {
                        "session_id": st.session_state.get("session_id", "dynamic_streamlit_user"),
                        "user_query": user_query,
                        "bot_response": response_text
                    }

                    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
                    supabase.table("chat_logs").insert(log_data).execute()

                except Exception as e:
                    st.error(f"Execution boundary interrupted: {e}")
else:
    st.warning("👈 Please supply a PDF document and build the knowledge index in the sidebar to open the query frame.")
