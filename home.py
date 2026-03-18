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
if "chat_history" not in st.session_state:
    st.session_state.chat_history=[("ai","Hello 👋, How may I assist you!")]

# embedding model
embedding=OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# Load Vector Db
@st.cache_resource
def load_vector_db():
    return Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding
    )
vector_store=load_vector_db()
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

# display chat history
for role,message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)

# User Input
user_input=st.chat_input("Ask Something...")
if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    start_time=time.time()
    with st.spinner("Searching documents..... 🤔"):
        response=rag_chain.invoke({
            "question":user_input,
            "chat_history":[]
        })
    end_time=time.time()
    total_time=end_time-start_time
    # Show bot message
    with st.chat_message("Assistant"):
        placeholder=st.empty()
        full_text=""
        for chunk in response:
            full_text+=chunk
            placeholder.markdown(full_text)
        st.write(f"Time Taken : {total_time:.2f} seconds")

    # Update Memory
    st.session_state.chat_history.append(("human",user_input))
    st.session_state.chat_history.append(("ai",response))