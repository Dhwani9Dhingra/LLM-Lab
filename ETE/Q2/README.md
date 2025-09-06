# Domain Specific Model:Legal ChatBot
LawGPT is a Large Language Model (LLM) based chatbot designed to provide legal information. The chatbot utilizes RAG architecture, advanced language models and embeddings to retrieve and generate contextually relevant answers from a provided legal document corpus. This project specifically focuses on the Indian Penal Code and other related legal documents.

## Features

- Conversational interface for querying legal information
- Uses FAISS for efficient vector search
- Embeds documents using Google Generative AI Embeddings
- Handles large document sets by splitting and batching
- Provides sources for retrieved information

## Tools and Technology
-Langchain
-LLama 3.3-70b
-FAISS Database
-Google Generative Embeddings
-ConversationBufferWindowMemory


## Architecture

The architecture of LawGPT includes the following components:

1. **Document Loader**: Loads legal documents from a directory of PDF files.
2. **Text Splitter**: Splits documents into manageable chunks for embedding.
3. **Embeddings**: Uses Google Generative AI Embeddings to transform text into vector representations.
4. **Vector Store**: Utilizes FAISS to store and retrieve document embeddings.
5. **LLM**: Uses the ChatGroq API to generate responses based on retrieved documents and user queries.
6. **Memory**: Maintains a conversation buffer to provide context in conversations.



```




