from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import streamlit as st
import time

# Title
st.set_page_config(page_title="RAG Chatbot",page_icon="🤖")
st.title("🤖 RAG CHATBOT ASSISTANT")

# embedding model
embedding=OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# Load Vector Db
@st.cache_resource
def load_models():
    vector_store = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding
    )

    retriever=vector_store.as_retriever(
        search_type="mmr",              #Maximal Marginal Relevance
        search_kwargs={"k":3}
    )

    # LLM Model
    llm=ChatOllama(
        model="llama3.2",
        base_url="http://localhost:11434",
        temperature=0.1,
        num_predict=300,
        streaming=True
    )

    return retriever, llm

retriever, llm=load_models()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello 👋, How may I assist you!"}
    ]

# Render chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# Prompt for final answer
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a AI assistant, you need to answer the question as detail as possible from the provide context,if the answer is not available in the provided context, just say I dont know, dont provide wrong answer.\n\n"
        "Context: \n{context}\n"),
        MessagesPlaceholder("chat_history"),
        ("human","{question}")
    ]
)

# Function to format document
def format_docs(docs):
    return("\n".join([doc.page_content for doc in docs]))

# Rag chain
rag_chain=(
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history")
    }
    | prompt | llm | StrOutputParser()
)

user_input = st.chat_input("Ask Something...")

if user_input:
    # Show user message instantly
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    # Assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""

        start_time = time.time()
        placeholder.markdown("Thinking...")
        # streaming response
        for chunk in rag_chain.stream({
            "question": user_input,
            "chat_history": []
        }):
            full_text += chunk
            placeholder.markdown(full_text + "▌")

        end_time = time.time()
        total_time = end_time - start_time

        placeholder.markdown(full_text)
        st.caption(f"⏱️ Time Taken: {total_time:.2f} sec")

    # Save response
    st.session_state.chat_history.append(
        {"role": "assistant", "content": full_text}
    )