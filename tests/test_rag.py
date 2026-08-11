import pytest

from rag import ingest_docs, retrieve


@pytest.fixture(scope="module", autouse=True)
def seeded_docs():
    # Real ChromaDB + the real docs/ dir, same as production — no mocking.
    # ingest_docs() rebuilds the collection from scratch, so this is safe to
    # run regardless of what other tests or local runs left behind.
    ingest_docs()


def test_retrieve_returns_relevant_docs_for_a_matching_query():
    docs = retrieve("what is your return policy")
    assert docs
    assert any("return" in d.lower() for d in docs)


def test_retrieve_filters_out_irrelevant_results():
    # Neither of these has anything to do with the seeded company docs
    # (shipping/returns/warranty/leave/expense/remote-work/product FAQ) —
    # the distance-based filter should leave nothing to return.
    assert retrieve("write a react login component") == []
    assert retrieve("what is the capital of france") == []


def test_retrieve_handles_hinglish_query():
    docs = retrieve("leave policy kya hai")
    assert docs
    assert any("leave" in d.lower() for d in docs)


def test_ingest_docs_splits_multi_paragraph_files_into_separate_chunks():
    # product_faq.txt has one paragraph per product plus a title and a
    # troubleshooting paragraph — a query about one product shouldn't pull
    # back an unrelated product's own dedicated paragraph too. (The shared
    # troubleshooting paragraph mentioning multiple products by name is a
    # separate chunk and is fine to include.)
    docs = retrieve("is the wireless mouse compatible with mac")
    assert any("wireless mouse" in d.lower() for d in docs)
    assert not any("monitor 27in ($249.99)" in d.lower() for d in docs)
