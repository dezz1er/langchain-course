import asyncio
import os
import ssl
from typing import Any

import certifi
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_tavily import TavilyExtract, TavilyMap
from langchain_community.embeddings import HuggingFaceEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from logger import (Colors, log_error, log_info, log_success, log_header,log_warning)


load_dotenv()


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "langchain_docs"

CHUNK_SIZE = 4000
CHUNK_OVERLAP = 400

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
qdrant_client = QdrantClient(url=QDRANT_URL)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={
        "batch_size": 32,
        "normalize_embeddings": True,
    },
)


ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


def ensure_collection() -> None:
    """
    Создаёт коллекцию в Qdrant, если её ещё нет.
    Размерность вектора определяем по одному тестовому embedding.
    """
    log_header("QDRANT COLLECTION INIT")

    vector_size = EMBEDDING_DIM
    collections = qdrant_client.get_collections().collections
    collection_names = {c.name for c in collections}

    if COLLECTION_NAME in collection_names:
        log_info(
            f"Qdrant: Collection '{COLLECTION_NAME}' already exists",
            Colors.DARKCYAN,
        )
        return

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    log_success(
        f"Qdrant: Collection '{COLLECTION_NAME}' created with vector size {vector_size}"
    )

def filter_docs_urls(urls: list[str]) -> list[str]:
    """
    Оставляем только документацию LangChain (Python).
    """
    return list({
        url for url in urls
        if "docs.langchain.com/oss/python" in url
    })

def get_vectorstore() -> QdrantVectorStore:
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

def chunk_urls(urls: list[str], chunk_size: int = 5) -> list[list[str]]:
    return [
        urls[i:i + chunk_size]
        for i in range(0, len(urls), chunk_size)
    ]

async def extract_batch(urls: list[str], batch_num: int) -> list[dict[str, Any]]:
    try:
        log_info(f"🔄 Extracting batch {batch_num} ({len(urls)} urls)", Colors.BLUE)
        res = await tavily_extract.ainvoke({"urls": urls})
        results = res.get("results", [])

        log_success(
            f"Batch {batch_num}: extracted {len(results)} pages"
        )
        return results

    except Exception as e:
        log_error(f"Batch {batch_num} failed: {e}")
        return []

def to_documents(results: list[dict[str, Any]]) -> list[Document]:
    docs: list[Document] = []

    for item in results:
        content = clean_text(item.get("raw_content", ""))
        url = item.get("url", "")

        if not content or len(content) < 200:
            continue

        docs.append(
            Document(
                page_content=content,
                metadata={"source": url},
            )
        )

    return docs

def chunk_documents(documents: list[Document]) -> list[Document]:
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️ Text Splitter: Processing {len(documents)} documents "
        f"with {CHUNK_SIZE} chunk size and {CHUNK_OVERLAP} overlap",
        Colors.YELLOW,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    split_docs = text_splitter.split_documents(documents)

    log_success(
        f"Text Splitter: Created {len(split_docs)} chunks from {len(documents)} documents"
    )
    return split_docs

import re

def clean_text(text: str) -> str:
    # remove markdown ![...](...)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    # remove links, keep text
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)

    # remove extra characters
    text = re.sub(r"`{3,}.*?`{3,}", "", text, flags=re.DOTALL)

    # remove single `code`
    text = re.sub(r"`([^`]*)`", r"\1", text)

    # remove extra whitespace
    text = re.sub(r"\n\s*\n", "\n\n", text)

    return text.strip()

async def index_documents_async(documents: list[Document], batch_size: int = 64) -> None:
    """
    Индексация документов в Qdrant батчами.
    Сам клиент синхронный, поэтому используем asyncio.to_thread.
    """
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 Qdrant Indexing: Preparing to add {len(documents)} documents",
        Colors.DARKCYAN,
    )

    vectorstore = get_vectorstore()
    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

    log_info(
        f"📦 Qdrant Indexing: Split into {len(batches)} batches "
        f"of up to {batch_size} documents each"
    )

    async def add_batch(batch: list[Document], batch_num: int) -> bool:
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch "
                f"{batch_num}/{len(batches)} ({len(batch)} documents)"
            )
            return True
        except Exception as exc:
            log_error(
                f"VectorStore Indexing: Failed to add batch {batch_num} - {exc}"
            )
            return False

    results = await asyncio.gather(
        *(add_batch(batch, i + 1) for i, batch in enumerate(batches)),
        return_exceptions=True,
    )

    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! "
            f"({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )



async def main() -> None:
    log_header("DOCUMENTATION INGESTION PIPELINE")

    ensure_collection()

    base_url = "https://docs.langchain.com/oss/python/langchain/"

    log_info("🗺️ TavilyMap: mapping docs structure...", Colors.PURPLE)

    site_map = tavily_map.invoke(base_url)
    urls = site_map.get("results", [])

    log_info(f"Discovered {len(urls)} URLs")

    urls = filter_docs_urls(urls)
    log_info(f"After filtering: {len(urls)} URLs")

    urls = list(set(urls))
    urls = urls[:200]

    log_info(f"Processing {len(urls)} URLs total")

    url_batches = chunk_urls(urls, chunk_size=5)

    log_info(f"Split into {len(url_batches)} batches")

    tasks = [
        extract_batch(batch, i + 1)
        for i, batch in enumerate(url_batches)
    ]

    batch_results = await asyncio.gather(*tasks)

    all_results: list[dict[str, Any]] = []
    for batch in batch_results:
        all_results.extend(batch)

    log_success(f"Extracted {len(all_results)} pages total")

    documents = to_documents(all_results)

    log_success(f"Converted to {len(documents)} documents")

    split_docs = chunk_documents(documents)

    await index_documents_async(split_docs, batch_size=20)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")

    log_info(f"   • URLs processed: {len(urls)}")
    log_info(f"   • Documents: {len(documents)}")
    log_info(f"   • Chunks: {len(split_docs)}")


if __name__ == "__main__":
    asyncio.run(main())