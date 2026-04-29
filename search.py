from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


client = QdrantClient(url="http://localhost:6333")
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="langchain_docs",
    embedding=embeddings,
)

docs = vectorstore.similarity_search(
    "How do I build a RAG application with LangChain?",
    k=3,
)

for doc in docs:
    print(doc.metadata)
    print(doc.page_content)
    print("=" * 80)