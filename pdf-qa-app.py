# main.py
import os
import streamlit as st
import sqlite3
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import hashlib

# Initialize database
def init_db():
    conn = sqlite3.connect('pdf_qa.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        doc_name TEXT NOT NULL,
        embedding_path TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    conn.commit()
    conn.close()

# User authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password):
    conn = sqlite3.connect('pdf_qa.db')
    c = conn.cursor()
    try:
        hashed_pw = hash_password(password)
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect('pdf_qa.db')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result and result[0] == hash_password(password):
        return True
    return False

def get_user_id(username):
    conn = sqlite3.connect('pdf_qa.db')
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

# PDF processing functions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks, user_id, doc_name):
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    
    # Save the embeddings
    embedding_path = f"embeddings/user_{user_id}_{doc_name.replace(' ', '_')}"
    vectorstore.save_local(embedding_path)
    
    # Save reference in database
    conn = sqlite3.connect('pdf_qa.db')
    c = conn.cursor()
    c.execute("INSERT INTO documents (user_id, doc_name, embedding_path) VALUES (?, ?, ?)", 
              (user_id, doc_name, embedding_path))
    conn.commit()
    conn.close()
    
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.5})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def get_user_documents(user_id):
    conn = sqlite3.connect('pdf_qa.db')
    c = conn.cursor()
    c.execute("SELECT id, doc_name, embedding_path FROM documents WHERE user_id = ?", (user_id,))
    docs = c.fetchall()
    conn.close()
    return docs

def load_vectorstore(embedding_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(embedding_path, embeddings)
    return vectorstore

def main():
    init_db()
    
    st.set_page_config(page_title="PDF Question Answering App", page_icon="📚")
    
    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Authentication
    if not st.session_state.authenticated:
        st.title("PDF Question Answering App")
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                login_button = st.form_submit_button("Login")
                
                if login_button:
                    if verify_user(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.user_id = get_user_id(username)
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
        
        with tab2:
            with st.form("signup_form"):
                new_username = st.text_input("Choose Username")
                new_password = st.text_input("Choose Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                signup_button = st.form_submit_button("Sign Up")
                
                if signup_button:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters long")
                    elif create_user(new_username, new_password):
                        st.success("Account created successfully! Please log in.")
                    else:
                        st.error("Username already exists")
    
    else:
        # Main application
        st.title("PDF Question Answering")
        st.write(f"Welcome, {st.session_state.username}!")
        
        if st.button("Logout"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
        
        # Sidebar
        with st.sidebar:
            st.subheader("Your Documents")
            docs = get_user_documents(st.session_state.user_id)
            
            if docs:
                selected_doc = st.selectbox(
                    "Select a document",
                    options=[doc[1] for doc in docs],
                    format_func=lambda x: x
                )
                
                if st.button("Load Selected Document"):
                    doc_path = next((doc[2] for doc in docs if doc[1] == selected_doc), None)
                    if doc_path:
                        with st.spinner("Loading document..."):
                            vectorstore = load_vectorstore(doc_path)
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                            st.success(f"Document '{selected_doc}' loaded successfully!")
                            st.session_state.chat_history = []
            
            st.subheader("Upload New Document")
            pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True, type="pdf")
            doc_name = st.text_input("Document Name")
            
            if st.button("Process"):
                if pdf_docs and doc_name:
                    with st.spinner("Processing..."):
                        # Get PDF text
                        raw_text = get_pdf_text(pdf_docs)
                        
                        # Get text chunks
                        text_chunks = get_text_chunks(raw_text)
                        
                        # Create vector store
                        vectorstore = get_vectorstore(text_chunks, st.session_state.user_id, doc_name)
                        
                        # Create conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.session_state.chat_history = []
                        
                        st.success("Document processed successfully!")
                else:
                    st.warning("Please upload PDF files and provide a document name")
        
        # Chat interface
        st.subheader("Ask questions about your document")
        
        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(f"User: {message}")
            else:
                st.write(f"AI: {message}")
        
        user_question = st.text_input("Ask a question about your PDF:")
        
        if user_question and st.session_state.conversation:
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({
                    "question": user_question,
                    "chat_history": [(st.session_state.chat_history[i], st.session_state.chat_history[i+1]) 
                                   for i in range(0, len(st.session_state.chat_history), 2) if i+1 < len(st.session_state.chat_history)]
                })
                
                st.session_state.chat_history.append(user_question)
                st.session_state.chat_history.append(response["answer"])
                
                st.rerun()
        elif user_question:
            st.info("Please process a document first")

if __name__ == "__main__":
    main()
