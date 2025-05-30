import streamlit as st
import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.messages import AIMessage # To handle LLM response object

from config import GROQ_API_KEY, MODEL_NAME, EMBEDDING_MODEL_NAME, MAX_RETRIES as CONFIG_MAX_RETRIES

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state: # To store vectorstore globally for chat
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None


st.set_page_config(page_title="Corrective RAG Chatbot")
st.title("🔍 Corrective RAG Chatbot")

# Custom Prompt for Correction (used by qa_chain)
corrective_prompt = PromptTemplate.from_template(
    """You are an AI assistant tasked with answering questions based on provided context.
    Always validate the answer against the context. If the answer is not found, say so.

    Question: {question}
    Context: {context}
    Answer:"""
)

# Load LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)

# --- Functions (evaluate_relevance, modify_query) ---
def evaluate_relevance(query: str, documents: list, llm_eval: ChatGroq) -> dict:
    if not documents:
        return {"relevant_enough": False, "reason": "No documents were retrieved."}
    doc_summaries = "\n".join([f"Doc {i+1}: {doc.page_content[:200]}..." for i, doc in enumerate(documents)])
    prompt_text = f"""Query: "{query}"
    Context (summaries):
    {doc_summaries}
    Are the provided document summaries relevant enough to answer the query? Respond 'yes' or 'no', then optionally a brief reason if 'no'.
    """
    try:
        evaluation = llm_eval.invoke(prompt_text)
        eval_text = (evaluation.content if isinstance(evaluation, AIMessage) else str(evaluation)).lower().strip()
        if eval_text.startswith("yes"):
            return {"relevant_enough": True, "reason": ""}
        else:
            reason = eval_text.replace("no", "").strip().lstrip(",.").strip()
            return {"relevant_enough": False, "reason": reason if reason else "Documents deemed irrelevant."}
    except Exception as e:
        print(f"Relevance evaluation error: {e}")
        return {"relevant_enough": False, "reason": f"Exception: {str(e)}"}

def modify_query(original_query: str, feedback_reason: str, llm_eval: ChatGroq) -> str:
    prompt_text = f"""Original query: "{original_query}" led to irrelevant docs (Reason: "{feedback_reason}").
    Generate a revised, more specific query to find better info. Return only the query string.
    Revised Query:"""
    try:
        response_llm = llm_eval.invoke(prompt_text) # Renamed to avoid conflict with qa_chain response
        revised_query = (response_llm.content if isinstance(response_llm, AIMessage) else str(response_llm)).strip()
        return revised_query.strip('"') 
    except Exception as e:
        print(f"Query modification error: {e}")
        return original_query


# --- File Upload and RAG Setup ---
uploaded_file = st.file_uploader(
    "Upload a PDF, TXT, or DOCX file to chat with", 
    type=["pdf", "txt", "docx"],
    key="file_uploader_key" 
)

if uploaded_file:
    file_name = uploaded_file.name
    temp_dir = "temp_uploads"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, file_name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    docs = []
    loader = None
    try:
        st.info(f"Processing '{file_name}'...")
        file_extension = os.path.splitext(file_name)[1].lower()

        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".txt":
            loader = TextLoader(file_path)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(file_path)
        else:
            st.error(f"Unsupported file type: {file_extension}. Please upload a PDF, TXT, or DOCX file.")
            docs = [] 

        if loader:
            docs = loader.load_and_split()
            st.success(f"Successfully loaded and split '{file_name}'. {len(docs)} document chunks found.")
        
        if not docs: 
            st.warning(f"No content could be extracted from '{file_name}'.")
            st.session_state.vectorstore = None 
            st.session_state.qa_chain = None
        else:
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": corrective_prompt},
                return_source_documents=True
            )
            st.success("Document processed and chatbot is ready!")

    except Exception as e:
        st.error(f"Error processing file '{file_name}': {e}")
        docs = [] 
        st.session_state.vectorstore = None
        st.session_state.qa_chain = None
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                st.warning(f"Could not delete temporary file {file_path}: {e}")
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            try:
                os.rmdir(temp_dir)
            except Exception as e:
                st.warning(f"Could not remove temporary directory {temp_dir}: {e}")


# --- Chat Interface ---
# Display chat history in the sidebar
st.sidebar.header("Chat History")
for message in st.session_state.chat_history:
    with st.sidebar.chat_message(message["role"]):
        st.markdown(message["content"]) # Display main content
        # If sources were stored with the message, display them too (optional enhancement)
        if "sources" in message:
             with st.sidebar.expander("Sources"):
                for src in message["sources"]:
                    st.caption(src)


if st.session_state.qa_chain:
    user_input = st.chat_input("Ask your question about the document...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"): # User message in main column
            st.markdown(user_input)

        current_query = user_input
        # MAX_RETRIES = 1 # Now using CONFIG_MAX_RETRIES
        response_generated = False
        assistant_response_content = ""
        assistant_response_sources_for_history = []


        with st.chat_message("assistant"): # Assistant message in main column
            if not st.session_state.vectorstore: 
                st.error("Chatbot is not ready. Please upload and process a document first.")
                assistant_response_content = "Chatbot is not ready. Please upload and process a document first."
                response_generated = True 
            else:
                current_retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

                for attempt in range(CONFIG_MAX_RETRIES + 1):
                    # UI element for attempt count - good for user feedback
                    # st.write(f"Attempt {attempt + 1}/{CONFIG_MAX_RETRIES + 1} for query: \"{current_query}\"") 
                    
                    retrieved_docs = current_retriever.get_relevant_documents(current_query)

                    if not retrieved_docs:
                        st.warning("No documents were retrieved for the current query.")
                        if attempt < CONFIG_MAX_RETRIES:
                            current_query = modify_query(current_query, "No documents found, query might be too specific or off-topic.", llm)
                            st.info(f"No documents found. Retrying with modified query: \"{current_query}\"") # st.info for less alarming message
                            continue
                        else:
                            assistant_response_content = "Failed to retrieve documents after multiple attempts."
                            st.error(assistant_response_content)
                            response_generated = True
                            break
                    
                    relevance_feedback = evaluate_relevance(current_query, retrieved_docs, llm)

                    if relevance_feedback["relevant_enough"]:
                        st.success("Relevant documents found. Generating answer...")
                        response = st.session_state.qa_chain.invoke(current_query) 
                        assistant_response_content = response["result"]
                        st.markdown(assistant_response_content)
                        
                        # --- Display Source Documents ---
                        if 'source_documents' in response and response['source_documents']:
                            assistant_response_sources_for_history = []
                            with st.expander("Show sources"):
                                for i, doc in enumerate(response['source_documents']):
                                    st.markdown(f"**Source {i+1}**")
                                    source_info = []
                                    if 'source' in doc.metadata:
                                        source_info.append(f"File: {os.path.basename(doc.metadata['source'])}")
                                    if 'page' in doc.metadata: 
                                        source_info.append(f"Page: {doc.metadata['page'] + 1}")
                                    
                                    source_caption = ", ".join(source_info)
                                    st.caption(source_caption)
                                    assistant_response_sources_for_history.append(source_caption + f"\nContent: {doc.page_content[:100]}...")


                                    st.markdown(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                    if i < len(response['source_documents']) - 1:
                                        st.divider()
                        # --- End Display Source Documents ---
                        response_generated = True
                        break 
                    else: 
                        st.warning(f"Documents found were not sufficiently relevant. Reason: {relevance_feedback.get('reason', 'N/A')}")
                        if attempt < CONFIG_MAX_RETRIES:
                            current_query = modify_query(current_query, relevance_feedback["reason"], llm)
                            st.info(f"Retrying with a modified query: \"{current_query}\"") # st.info for less alarming message
                        else:
                            assistant_response_content = f"Sorry, I could not find sufficiently relevant information for \"{user_input}\" after {CONFIG_MAX_RETRIES + 1} attempts. Last reason: {relevance_feedback.get('reason', 'N/A')}"
                            st.error(assistant_response_content)
                            response_generated = True
                            break
                
                if not response_generated:
                     assistant_response_content = "An unexpected error occurred, and no response was generated."
                     st.error(assistant_response_content)
            
            # Add assistant response to chat history (main content and sources)
            history_entry = {"role": "assistant", "content": assistant_response_content}
            if assistant_response_sources_for_history:
                history_entry["sources"] = assistant_response_sources_for_history
            st.session_state.chat_history.append(history_entry)

elif uploaded_file and not st.session_state.qa_chain : 
    st.error("Chatbot initialization failed. Please check the file or try a different one.")
else: 
    st.info("Please upload a document (PDF, TXT, or DOCX) to begin the chat.")
