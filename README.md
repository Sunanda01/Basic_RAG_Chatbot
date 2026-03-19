# 🤖 Basic RAG Chatbot

A simple **Retrieval-Augmented Generation (RAG) Chatbot** built using LangChain, Ollama, ChromaDB, and Streamlit.

This chatbot answers user queries based on document context using vector search + LLM reasoning.

---

## 🚀 Features

- 🔍 Semantic search using vector embeddings
- 🧠 Context-aware answers using RAG pipeline
- 💬 Conversational interface with chat history
- ⚡ Local LLM using Ollama (no API cost)
- 📊 Fast retrieval using Chroma vector database
- 🎯 Accurate answers grounded in provided documents

---

## 🏗️ Tech Stack

- **Backend Framework:** LangChain  
- **LLM:** Ollama (LLaMA 3.2)  
- **Embeddings:** Ollama Embeddings (`nomic-embed-text`)  
- **Vector Database:** ChromaDB  
- **Frontend:** Streamlit  
- **Language:** Python  

---

## ⚙️ How It Works (RAG Flow)

1. User enters a question  
2. Query is converted into embeddings  
3. ChromaDB retrieves relevant document chunks  
4. Context is combined with the question  
5. LLM (Ollama) generates final answer  

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/Sunanda01/Basic_RAG_Chatbot.git
cd Basic_RAG_Chatbot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install & Run Ollama
Run required models:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 5. Run the application
```bash
# Document Injestion 
python injest.py

# Run UI
streamlit run home.py
```

---

## 💬 Usage
```bash
Open browser at http://localhost:8501
Ask questions based on your documents
Chatbot will respond using retrieved context
```

---

## 📸 UI

<img width="953" height="722" alt="Screenshot 2026-03-19 010553" src="https://github.com/user-attachments/assets/e9637531-347a-4cbd-8b2e-b9a923524d1e" />

---

## ▶️ Demo
[chatbot-demo.webm](https://github.com/user-attachments/assets/e78f6535-9458-4267-9a98-2cd8c0819963)

