import streamlit as st
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load JSON data and convert it into text
def get_json_text(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    text = json.dumps(data)
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save a vector store using FAISS
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Function to load the conversational chain for Q&A
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, say "The answer is not available in the context." Do not provide a wrong answer.
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and generate a response
def user_input(user_question, chat_history):
    # Load the conversational model for general questions
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    
    # Determine if the question is general or requires context from JSON files
    is_general = is_general_question(user_question)
    
    if is_general:
        chat_history.append(HumanMessage(content=user_question))
        response = model(chat_history)
        chat_history.append(AIMessage(content=response.content))
        response_text = response.content
    else:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question)
        
        if docs:
            # Load the conversational chain and get the response from the chain
            chain = get_conversational_chain()
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            response_text = response.get("output_text", None)
            
            # Fallback to general model if no answer is found
            if not response_text or "The answer is not available in the context" in response_text:
                chat_history.append(HumanMessage(content=user_question))
                response = model(chat_history)
                chat_history.append(AIMessage(content=response.content))
                response_text = response.content
        else:
            # Direct fallback to the conversational model for general questions
            chat_history.append(HumanMessage(content=user_question))
            response = model(chat_history)
            chat_history.append(AIMessage(content=response.content))
            response_text = response.content
    
    return response_text, chat_history

# Improved general question detection
def is_general_question(question):
    general_keywords = ["who", "what", "where", "when", "how", "why", "can", "could", "would"]
    return any(keyword in question.lower() for keyword in general_keywords)

def main():
    st.set_page_config(page_title="Chat Bot")
    st.header("Question & Answer Agent")

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a Question about Movies, Books, or general topics.")
    
    if user_question:
        # Handle both general and specific questions
        response, st.session_state.chat_history = user_input(user_question, st.session_state.chat_history)
        st.write("Reply:", response)

    # Display chat history
    st.subheader("Conversation History")
    for i, message in enumerate(st.session_state.chat_history):
        if isinstance(message, HumanMessage):
            st.write(f"Q{i//2 + 1}: {message.content}")
        elif isinstance(message, AIMessage):
            st.write(f"A{i//2 + 1}: {message.content}")

    with st.sidebar:
        st.title("Upload Documents")
        st.write("The system will automatically use the latest data from the uploaded documents.")
        if st.button("Process Documents"):
            with st.spinner("Processing..."):
                all_text = ""
                base_path = "./data/"  # Directory where JSON files are stored
                for filename in os.listdir(base_path):
                    file_path = os.path.join(base_path, filename)
                    if filename.endswith(".json"):
                        all_text += get_json_text(file_path)
                text_chunks = get_text_chunks(all_text)
                get_vector_store(text_chunks)
                st.success("Processing complete. You can now ask questions.")

if __name__ == "__main__":
    main()
