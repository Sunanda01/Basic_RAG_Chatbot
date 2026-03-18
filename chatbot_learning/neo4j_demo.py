# Neo4j Configuration
# 1. Store Embedding in Neo4j
"""
    Neo4jVector.from_documents(
        documents=text_chunks,
        embedding=embedding,
        url=NEO4J_URL,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name=NEO4J_INDEX_NAME
    )
"""

# 2. Load Vector Store
"""
    Neo4jVector.from_existing_index(
        embedding=embedding,
        url=NEO4J_URL,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name=NEO4J_INDEX_NAME
    )
"""

"""
    We can also small LLAMA models to get more response in less time
        Llama LLM Model => llama3.2:3b
        Llama Embedding Model => mxbai-embed-large
        base_url => http://localhost:11434
"""
