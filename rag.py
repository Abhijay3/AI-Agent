import glob
import os

import chromadb

COLLECTION_NAME = "company_docs"
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

# ChromaDB's default distance metric here is squared L2 (lower = more
# relevant). Measured empirically against this doc set: genuinely relevant
# matches score <=~1.5, unrelated queries (coding questions, small talk,
# other topics) score >=~1.7. Anything above the cutoff is noise, not
# useful context — including it just dilutes the prompt and risks the
# model treating unrelated policy text as relevant.
MAX_RELEVANT_DISTANCE = 1.5


def _chunk_document(text: str) -> list:
    # Split on blank lines (paragraph breaks). Small single-topic docs
    # (e.g. the policy files) naturally stay as one chunk; multi-topic docs
    # (e.g. the product FAQ, one paragraph per product) split into chunks
    # that can be retrieved independently, so a question about one product
    # doesn't drag in irrelevant ones.
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def ingest_docs(docs_dir: str = "docs") -> None:
    paths = glob.glob(os.path.join(docs_dir, "*.txt"))

    ids = []
    documents = []
    for path in paths:
        with open(path, "r") as f:
            chunks = _chunk_document(f.read())
        basename = os.path.basename(path)
        for i, chunk in enumerate(chunks):
            ids.append(f"{basename}#{i}")
            documents.append(chunk)

    # Recreate the collection instead of upserting in place: chunk IDs are
    # derived from position, so if a doc's paragraph count ever shrinks,
    # upsert alone would leave stale chunks from the old, longer version
    # behind. A clean rebuild is cheap for a doc set this size and keeps
    # ingestion idempotent regardless of how the source files change.
    global collection
    chroma_client.delete_collection(COLLECTION_NAME)
    collection = chroma_client.create_collection(COLLECTION_NAME)
    collection.upsert(ids=ids, documents=documents)
    print(f"Ingested {len(paths)} documents into {len(ids)} chunks")


def retrieve(query: str, n_results: int = 3) -> list:
    if collection.count() == 0:
        return []
    results = collection.query(query_texts=[query], n_results=n_results)
    docs = results["documents"][0]
    distances = results["distances"][0]
    return [doc for doc, dist in zip(docs, distances) if dist <= MAX_RELEVANT_DISTANCE]


if __name__ == "__main__":
    ingest_docs()
