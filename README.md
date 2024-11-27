# **Custom Data Chatbot Using RAG (Retrieval-Augmented Generation)**

![Chatbot Banner](https://nordvpn.com/wp-content/uploads/blog-featured-what-is-chatbot.svg)

---

## **Overview**
This repository demonstrates a **Custom Data Chatbot** built using **Retrieval-Augmented Generation (RAG)**. The project combines **document retrieval** and **language model capabilities** to answer queries from user-supplied PDF documents. This system is ideal for applications such as customer support, knowledge base automation, and intelligent document query systems.

---

## **Problem Statement**
Traditional keyword-based search systems often fail to provide precise and context-aware results. For example:
- Businesses may struggle to retrieve specific answers from dense documents such as manuals or policies.
- Users may require detailed explanations or summaries rather than just locating keywords.
- 
![RAG Banner](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*d4XRH4R7p40_vzD6CFYXsQ.jpeg)

### **Objective**
To develop a chatbot that:
1. Processes large PDF documents into manageable and searchable chunks.
2. Utilizes state-of-the-art semantic search techniques for retrieving relevant information.
3. Provides detailed, accurate, and context-aware answers using a **local language model**.

---

## **Features**
### **Key Functionalities**
1. **Load and preprocess PDF documents**: Extract data from multiple PDFs and prepare it for analysis.
2. **Semantic indexing**: Use **sentence-transformers** for embedding and similarity-based search.
3. **Persistent database**: Leverage **ChromaDB** for efficient storage and retrieval.
4. **Context-aware answers**: Generate accurate responses using fine-tuned LLMs.
5. **Customizable queries**: Tailor the bot to specific datasets or industries.

---

## **Architecture**

### **Workflow**
1. **Document Loading**: Extract and preprocess data from PDFs.
2. **Chunking**: Split large text into smaller pieces for efficient handling.
3. **Embedding**: Convert text into vector representations using Hugging Face models.
4. **Database Storage**: Save embeddings into a **persistent ChromaDB** collection.
5. **Query Handling**:
   - Retrieve relevant documents based on user input.
   - Use context from retrieved documents to generate accurate responses.

---

## **Installation and Setup**
### **Prerequisites**
- **Python 3.8+**
- **Hugging Face Account and API Token**
- Required libraries: `langchain`, `chromadb`, `transformers`, `PyMuPDF`, `shutil`

### **Step-by-Step Instructions**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/custom-data-chatbot-RAG.git
   cd custom-data-chatbot-RAG
2. Install dependencies:

    Create a `requirements.txt` file with the following content:

    ```txt
    langchain
    transformers
    chromadb
    huggingface_hub
    PyMuPDF
    torch
    ```

    Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```
3. Log in to Hugging Face:

    You'll need to authenticate with Hugging Face for accessing models:

    ```python
    from huggingface_hub import login
    login()
    ```

---


## **Code Explanation**

### **Document Loading**

Extract data from PDF documents:

```python
def load_pdf(data_directory):
    loader = DirectoryLoader(data_directory, glob="*.pdf", loader_cls=PyMuPDFLoader)
    documents = loader.load()
    return documents
   ```
### **Text Splitting**

Divide the extracted text into manageable chunks:

```python
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks
 ```

### **Embedding and Indexing**

Store the processed chunks as embeddings in ChromaDB:

```python
db = Chroma.from_documents(text_chunks, embeddings, persist_directory=persist_directory, client=client)
 ```

### **Query Handling**

Answer user queries with the help of a local LLM:

```python
qa = RetrievalQA.from_chain_type(
    llm=local_llm,
    chain_type="stuff",
    retriever=db.as_retriever(),
    chain_type_kwargs=chain_type_kwargs
)
response = qa.run(query)
 ```

## **Sample Use Case**

### **Scenario**
1. **Document**: A user uploads a PDF document titled "The Thirsty Crow".
2. **Query**: "What is the crow looking for?"
3. **Bot Response**:"The crow is searching for water."


## **Future Improvements**

1. Add support for additional file types (e.g., Word, Excel).
2. Implement a web-based or mobile-friendly UI.
3. Enhance multi-language support.
4. Improve query understanding with advanced techniques like Reinforcement Learning with Human Feedback (RLHF).

## **License**
This project is licensed under the MIT License. See LICENSE for more information.

## **Acknowledgments**
1. **Hugging Face** for providing robust embeddings and LLMs.
2. **LangChain ** for simplifying chain-based workflows.
3. **ChromaDB ** for efficient and scalable vector storage.


