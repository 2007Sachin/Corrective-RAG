# Corrective RAG Chatbot

## Overview

This application is an interactive chatbot that leverages a corrective Retrieval Augmented Generation (RAG) pipeline. It allows users to upload documents (PDF, TXT, or DOCX) and ask questions based on their content. The chatbot employs a multi-step process to enhance the accuracy and relevance of its answers.

## Features

*   **Flexible File Uploads:** Supports PDF, TXT, and DOCX document formats.
*   **Corrective RAG Pipeline:**
    *   **Document Relevance Evaluation:** An LLM assesses if the initially retrieved document chunks are relevant to the user's query.
    *   **Query Modification & Re-retrieval:** If documents are deemed irrelevant, the system modifies the original query based on LLM feedback and attempts to retrieve more relevant documents. This process can be repeated up to a configurable number of retries.
*   **Display of Source Documents:** Shows snippets from the source documents used to generate the answer, promoting transparency.
*   **Interactive Chat Interface:** Provides a user-friendly chat interface built with Streamlit.

## Setup and Running

### Prerequisites

*   Python 3.7+

### 1. Clone the Repository

```bash
git clone <repository_url> # Replace <repository_url> with the actual URL
cd <repository_directory>
```

### 2. API Keys

This application requires a GROQ API key to function.

*   You will find a `config.py.example` file in the repository.
*   Create a copy of this file and name it `config.py`:
    ```bash
    cp config.py.example config.py
    ```
*   Open `config.py` and replace `"YOUR_GROQ_API_KEY_HERE"` with your actual Groq API key.

    ```python
    # config.py
    GROQ_API_KEY = "sk-your-actual-groq-api-key-goes-here" 
    # ... other configurations
    ```

### 3. Dependencies

Install the necessary Python packages using pip:

```bash
pip install -r requirements.txt
```
This will install all required libraries, including `langchain`, `streamlit`, `groq`, `faiss-cpu`, `pypdf`, `sentence-transformers`, and `docx2txt` (for DOCX file support).

*Self-correction note from development: If you encounter an 'OSError: [Errno 28] No space left on device' during `pip install`, please ensure your environment has sufficient disk space, as some dependencies (especially those related to PyTorch for sentence transformers) can be large.*

### 4. Running the Application

Once the dependencies are installed and the API key is set up, you can run the Streamlit application:

```bash
streamlit run app.py
```
This will typically open the application in your web browser.

## Configuration

Several aspects of the chatbot can be configured by editing the `config.py` file:

*   `GROQ_API_KEY`: Your API key for the Groq LLM service.
*   `MODEL_NAME`: The specific Groq LLM model to be used (e.g., `llama3-8b-8192`).
*   `EMBEDDING_MODEL_NAME`: The sentence-transformer model for creating document embeddings (e.g., `sentence-transformers/all-MiniLM-L6-v2`).
*   `MAX_RETRIES`: The maximum number of times the system will attempt to re-retrieve documents if the initial ones are not relevant (default is 1).

## Corrective RAG Concept

The core idea behind this application's Corrective RAG pipeline is to improve the reliability of answers by actively checking and refining the retrieval process. Standard RAG retrieves documents and generates an answer; this system adds layers of self-correction:

1.  **Initial Retrieval:** When a user asks a question, the system retrieves document chunks deemed relevant by semantic similarity.
2.  **Relevance Evaluation:** The retrieved chunks and the original query are presented to an LLM, which evaluates if these chunks are truly relevant for answering the query.
3.  **Conditional Re-Action:**
    *   **If Relevant:** The system proceeds to generate an answer using the validated relevant documents.
    *   **If Not Relevant:** The LLM provides a reason for the irrelevance. This reason is then used to modify the original query (e.g., by adding more specific keywords, rephrasing). The system then re-retrieves documents using this new, refined query. This loop can occur up to `MAX_RETRIES` times.
4.  **Answer Generation:** Once relevant documents are obtained (either initially or after re-retrieval), the final answer is generated based on this context.
5.  **Source Display:** The user is shown snippets from the source documents that contributed to the answer, allowing for verification.

This iterative refinement process helps in handling ambiguous queries or situations where initial document retrieval might not be optimal, leading to more accurate and contextually appropriate responses.
