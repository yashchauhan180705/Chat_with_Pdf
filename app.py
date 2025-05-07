# main.py
import os
import streamlit as st
import sqlite3
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import GPT4All
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import hashlib

# Set page configuration
st.set_page_config(
    page_title="PDF Question Answering App",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better UI
def load_css():
    st.markdown("""
    <style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message.user {
        background-color: #ffffff;
        border-left: 5px solid #7c4dff;
    }
    .chat-message.bot {
        background-color: #ffffff;
        border-left: 5px solid #2196f3;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex-grow: 1;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
        transition-duration: 0.4s;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .upload-btn {
        background-color: #2196F3;
    }
    .upload-btn:hover {
        background-color: #0b7dda;
    }
    .logout-btn {
        background-color: #f44336;
    }
    .logout-btn:hover {
        background-color: #d32f2f;
    }
    .st-emotion-cache-16txtl3 h1 {
        font-weight: 700;
        color: #1e3a8a;
    }
    .st-emotion-cache-16txtl3 h2 {
        font-weight: 600;
        color: #1e3a8a;
    }
    .app-header {
        text-align: center;
        padding: 1rem;
        background-color: #f8f9fa;
        border-bottom: 1px solid #e9ecef;
        margin-bottom: 2rem;
    }
    .pdf-icon {
        font-size: 48px;
        color: #ff5722;
    }
    .document-card {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        transition: all 0.3s;
    }
    .document-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        cursor: pointer;
    }
    .selected-document {
        border-left: 4px solid #4CAF50;
        background-color: #f1f8e9;
    }
    </style>
    """, unsafe_allow_html=True)


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


def setup_local_llm():
    # Check if model exists, if not, inform user to download it
    model_path = "ggml-gpt4all-j-v1.3-groovy.bin"

    if not os.path.exists(model_path):
        return None

    return GPT4All(model=model_path, max_tokens=512)


# Alternative approach using a simple retrieval method that doesn't require an LLM
def simple_qa(query, vectorstore):
    # Get relevant documents based on the query
    docs = vectorstore.similarity_search(query, k=3)

    # Construct response from the documents
    response = "Based on the documents, here's what I found:\n\n"
    for i, doc in enumerate(docs):
        response += f"**Excerpt {i + 1}**:\n{doc.page_content}\n\n"

    return response


def get_conversation_chain(vectorstore):
    # Try to set up local LLM
    llm = setup_local_llm()

    if llm is None:
        # If no LLM is available, return None to signal we should use fallback method
        return None

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
    # Fix for ValueError by setting allow_dangerous_deserialization to True
    vectorstore = FAISS.load_local(embedding_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore


def display_message(role, content):
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user">
            <div class="avatar">👤</div>
            <div class="message">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot">
            <div class="avatar">🤖</div>
            <div class="message">{content}</div>
        </div>
        """, unsafe_allow_html=True)


def main():
    init_db()
    load_css()

    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""
    if "processing_question" not in st.session_state:
        st.session_state.processing_question = False
    if "selected_doc" not in st.session_state:
        st.session_state.selected_doc = None

    # Authentication
    if not st.session_state.authenticated:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="app-header">
                <div class="pdf-icon">📚</div>
                <h1>PDF Question Answering App</h1>
                <p>Upload your PDF documents and ask questions about their content</p>
            </div>
            """, unsafe_allow_html=True)

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
        st.markdown(f"""
        <div class="app-header">
            <h1>PDF Question Answering</h1>
            <p>Welcome, {st.session_state.username}! Ask questions about your uploaded documents.</p>
        </div>
        """, unsafe_allow_html=True)

        # Layout with columns
        col1, col2 = st.columns([1, 3])

        # Sidebar (Documents and Upload)
        with col1:
            st.markdown("### Your Documents")

            # Logout button
            if st.button("Logout", key="logout_btn", help="Sign out from your account"):
                for key in st.session_state.keys():
                    del st.session_state[key]
                st.rerun()

            st.markdown("---")

            # Document list
            docs = get_user_documents(st.session_state.user_id)

            if docs:
                for doc_id, doc_name, doc_path in docs:
                    is_selected = st.session_state.selected_doc == doc_name
                    class_name = "document-card selected-document" if is_selected else "document-card"

                    st.markdown(f"""
                    <div class="{class_name}" id="doc-{doc_id}">
                        <strong>{doc_name}</strong>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button(f"Load", key=f"load_doc_{doc_id}"):
                        with st.spinner("Loading document..."):
                            try:
                                vectorstore = load_vectorstore(doc_path)
                                st.session_state.vectorstore = vectorstore
                                st.session_state.conversation = get_conversation_chain(vectorstore)
                                st.session_state.selected_doc = doc_name
                                st.success(f"Document '{doc_name}' loaded successfully!")
                                st.session_state.chat_history = []
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error loading document: {str(e)}")
            else:
                st.info("No documents uploaded yet.")

            st.markdown("---")

            # Upload section
            st.markdown("### Upload New Document")

            pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True, type="pdf")
            doc_name = st.text_input("Document Name")

            if st.button("Process Document", key="process_btn", help="Upload and process the selected PDF"):
                if pdf_docs and doc_name:
                    with st.spinner("Processing document..."):
                        try:
                            # Get PDF text
                            raw_text = get_pdf_text(pdf_docs)

                            # Get text chunks
                            text_chunks = get_text_chunks(raw_text)

                            # Create vector store
                            vectorstore = get_vectorstore(text_chunks, st.session_state.user_id, doc_name)

                            # Store vectorstore in session state
                            st.session_state.vectorstore = vectorstore
                            st.session_state.selected_doc = doc_name

                            # Create conversation chain
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                            st.session_state.chat_history = []

                            st.success("Document processed successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error processing document: {str(e)}")
                else:
                    st.warning("Please upload PDF files and provide a document name")

        # Main chat area
        with col2:
            if st.session_state.selected_doc:
                st.markdown(f"### Chatting with: **{st.session_state.selected_doc}**")
            else:
                st.markdown("### Select or Upload a Document to Start")

            # Chat container
            chat_container = st.container(height=400)
            with chat_container:
                if not st.session_state.chat_history:
                    st.info("No messages yet. Ask a question about your document!")

                for message in st.session_state.chat_history:
                    display_message(message["role"], message["content"])

            # Process any pending question from previous run
            if st.session_state.processing_question and st.session_state.user_question and st.session_state.vectorstore:
                user_question = st.session_state.user_question
                st.session_state.processing_question = False
                st.session_state.user_question = ""

                with st.spinner("Thinking..."):
                    try:
                        # Use the conversation chain if available, otherwise fall back to simple retrieval
                        if st.session_state.conversation:
                            # Create formatted chat history for the conversation chain
                            formatted_history = []
                            for i in range(len(st.session_state.chat_history)):
                                if st.session_state.chat_history[i]["role"] == "user" and i + 1 < len(
                                        st.session_state.chat_history):
                                    user_msg = st.session_state.chat_history[i]["content"]
                                    ai_msg = st.session_state.chat_history[i + 1]["content"]
                                    formatted_history.append((user_msg, ai_msg))

                            response = st.session_state.conversation({
                                "question": user_question,
                                "chat_history": formatted_history
                            })
                            answer = response["answer"]
                        else:
                            # Simple retrieval fallback method
                            answer = simple_qa(user_question, st.session_state.vectorstore)

                        # Append to chat history with role information
                        st.session_state.chat_history.append({"role": "user", "content": user_question})
                        st.session_state.chat_history.append({"role": "bot", "content": answer})

                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")

            # Question input area with improved UI
            st.markdown("---")

            # Only show input if a document is selected
            if st.session_state.selected_doc:
                col_input, col_button = st.columns([4, 1])

                with col_input:
                    user_question = st.text_input("Ask a question about your document:", key="question_input",
                                                  placeholder="e.g., What are the main topics discussed in this document?")

                with col_button:
                    submit_button = st.button("Ask", key="submit_question")

                if submit_button and user_question:
                    st.session_state.user_question = user_question
                    st.session_state.processing_question = True
                    st.rerun()
            else:
                st.info("Please select or upload a document first to ask questions.")

            # Display a note about model usage
            if not os.path.exists("ggml-gpt4all-j-v1.3-groovy.bin"):
                st.markdown("---")
                st.info(
                    "📝 Note: For better answers, download the GPT4All model from https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin and place it in the application directory.")


if __name__ == "__main__":
    main()