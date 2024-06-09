# Medical Chatbot

This repository contains a Python application that implements a conversational chatbot for medical assistance. The chatbot is built using the LangChain library and is powered by a pre-trained LLaMA (Large Language Model Adapted) model. The application ingests and processes medical documents, creates a vector store, and provides a Streamlit-based user interface for interacting with the chatbot.

## Features

- Ingests and processes PDF documents from a specified directory
- Splits the documents into smaller text chunks for efficient processing
- Creates embeddings for the text chunks using a HuggingFace sentence embeddings model
- Stores the text chunks and their embeddings in a FAISS vector store
- Utilizes a pre-trained LLaMA model for generating responses
- Maintains conversation context using a memory component
- Provides a text input field and an audio recorder for user queries
- Displays the chat history in the Streamlit application

## Directory Structure

- `app.py`: Contains the Streamlit application code for the chatbot interface.
- `ingest.py`: Handles the ingestion and preprocessing of PDF documents.
- `model.py`: Defines the main components of the chatbot, including the language model, embeddings, and the retrieval-based question-answering chain.
- `langchain.ipynb`: A Jupyter Notebook that demonstrates the core functionality of the chatbot in an interactive manner.
- `data/`: Directory containing the PDF documents to be ingested (not included in the repository).
- `vectorestores/`: Directory where the FAISS vector store is saved (created during runtime).

  # Code Details

## app.py

This file sets up the Streamlit application for the chatbot.

- Loads the FAISS vector store and embeddings from the local path.
- Creates an instance of the LLaMA language model using `CTransformers`.
- Initializes a `ConversationBufferMemory` to keep track of the chat history.
- Sets up the `ConversationalRetrievalChain` to handle the conversational flow, combining the language model, vector store retriever, and memory.
- Defines the Streamlit application with a text input field and an audio recorder for users to ask questions.
- When a question is submitted, calls the `conversation_chat` function to generate an answer using the `ConversationalRetrievalChain`.
- Displays the chat history in the Streamlit app using the `streamlit_chat` package.

## ingest.py

This file handles the ingestion and preprocessing of the data (PDF documents in the `data/` directory).

- Loads the PDF documents using the `DirectoryLoader` and `PyPDFLoader` from LangChain.
- Splits the loaded documents into smaller text chunks using the `RecursiveCharacterTextSplitter`.
- Generates embeddings for the text chunks using the HuggingFace embedding model.
- Creates a new FAISS vector store from the text chunks and their embeddings.
- Saves the vector store to the local path `vectorestores/db_faiss`.

## model.py

This file defines the main components of the chatbot, including the language model, embeddings, and the retrieval-based question-answering chain.

- Loads a pre-trained LLaMA model from a `.bin` file and uses the `CTransformers` class from LangChain to interface with the model.
- Creates embeddings using the HuggingFace sentence embeddings model `all-MiniLM-L6-v2`.
- Loads the FAISS vector store from the local path `vectorestores/db_faiss`.
- Defines a custom prompt template to format the query and context before passing them to the language model.
- Sets up the `RetrievalQA` chain to perform the question-answering task by retrieving relevant document chunks from the vector store and generating an answer using the language model.

## langchain.ipynb

This Jupyter Notebook demonstrates the core functionality of the chatbot in an interactive manner.

- Loads PDF documents from the `data/` directory using the `DirectoryLoader` and `PyPDFLoader`.
- Splits the loaded documents into smaller text chunks using the `RecursiveCharacterTextSplitter`.
- Generates embeddings for the text chunks using the `HuggingFaceEmbeddings` model.
- Creates a FAISS vector store from the text chunks and their embeddings, and saves it to the local path `vectorestores/db_faiss`.
- Defines a custom prompt template to format the query and context.
- Creates an instance of the LLaMA language model using `CTransformers`.
- Initializes a `ConversationBufferMemory` to keep track of the chat history.
- Sets up the `ConversationalRetrievalChain` using the language model, vector store retriever, and memory.
- Defines the `conversation_chat` function to handle user input and generate responses.
- Provides an interactive loop for users to ask questions and receive responses from the chatbot.

## Prerequisites

Before running the application, ensure that you have the following prerequisites installed:

- Python (version 3.7 or later)
- Required Python packages (listed in `requirements.txt`)
- LLaMA model files (e.g., `llama-2-7b-chat.ggmlv3.q8_0.bin`)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Idk507/health_bot
```

2. Navigate to the project directory:

```bash
cd health_bot
```
3.Install Requirements.txt
```bash
pip install -r requirements.txt
```

Download the ```LLaMA model files (e.g., llama-2-7b-chat.ggmlv3.q8_0.bin)``` and place them in the project directory.
Place the PDF documents you want to ingest in the data/ directory.

# Usage

Run the ingestion script to preprocess the documents and create the vector store:

 ```python ingest.py ```

Start the Streamlit application:

``` streamlit run app.py ```

The application will open in your default web browser, where you can interact with the chatbot by typing or recording your questions.

