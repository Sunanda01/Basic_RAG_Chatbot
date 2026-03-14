from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

# embedding model
embedding=OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# Load Vector Db
vector_store=Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

retriever=vector_store.as_retriever(search_kwargs={"k":3})

# LLM Model
llm=ChatOllama(
    model="llama3.2",
    base_url="http://localhost:11434",
    temperature=0.1
)

prompt_template="""
    You are a AI assistant, you need to answer the question as detail as possible from the provide context,if the answer is not available in the provided context, just say I dont know, dont provide wrong answer.
    Context: \n{context}\n
    Question: \n{question}\n
    Answer:
"""
while(True):
    query=input("Ask Question: ")
    if(query.lower()=='exit'):
        break
    docs=retriever.invoke(query)
    context="\n".join([doc.page_content for doc in docs])

    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])

    final_prompt=prompt.format(
        context=context,
        question=query
    )
    response = llm.invoke(final_prompt)
    print(response.content)