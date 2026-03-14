from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
# from dotenv import load_dotenv

# Load Document
loader=PyPDFDirectoryLoader("documents")
docs=loader.load()

# Split into chunks
text_splitter=RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=100)
text_chunks=text_splitter.split_documents(documents=docs)

# define embedding model
embedding=OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# Store the embedding in vectorDB => Chroma
vector_store=Chroma.from_documents(
    documents=text_chunks,
    embedding=embedding,
    persist_directory="chroma_db"
)
vector_store.persist()
print("Document embedded and stored in chromDB") 
