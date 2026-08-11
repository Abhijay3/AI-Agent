import glob
import os

import chromadb

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection("company_docs")


def ingest_docs(docs_dir: str = "docs") -> None:
    paths = glob.glob(os.path.join(docs_dir, "*.txt"))

    ids = []
    documents = []
    for path in paths:
        with open(path, "r") as f:
            documents.append(f.read())
        ids.append(os.path.basename(path))

    collection.upsert(ids=ids, documents=documents)
    print(f"Ingested {len(ids)} documents: {ids}")


def retrieve(query: str, n_results: int = 3) -> list:
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0]


if __name__ == "__main__":
    ingest_docs()
