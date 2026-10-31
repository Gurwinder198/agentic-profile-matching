"""Chroma-backed RAG search tool over the resume corpus (Milestone 2)."""

from langchain_chroma import Chroma

from llm import get_embeddings
from tools.filesystem_tools import list_resume_files, read_resume_file


def build_index(
    resumes_dir: str = "data/resumes",
    persist_dir: str = "data/chroma_db",
    force_rebuild: bool = False,
) -> Chroma:
    """Build (or load) the Chroma index over every resume in resumes_dir."""
    embeddings = get_embeddings()
    store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

    is_populated = len(store.get()["ids"]) > 0
    if is_populated and not force_rebuild:
        return store

    if is_populated and force_rebuild:
        existing_ids = store.get()["ids"]
        if existing_ids:
            store.delete(ids=existing_ids)

    filenames = list_resume_files(resumes_dir)
    texts = [read_resume_file(filename, resumes_dir) for filename in filenames]
    metadatas = [{"candidate_id": filename.removesuffix(".txt")} for filename in filenames]
    ids = [filename.removesuffix(".txt") for filename in filenames]
    if texts:
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return store


def rag_search_resumes(
    query: str,
    k: int = 10,
    resumes_dir: str = "data/resumes",
    persist_dir: str = "data/chroma_db",
) -> list[dict]:
    """Semantic-search the resume corpus, returning the top-k matches."""
    store = build_index(resumes_dir, persist_dir, force_rebuild=False)
    results = store.similarity_search_with_relevance_scores(query, k=k)
    output = []
    for document, score in results:
        candidate_id = document.metadata["candidate_id"]
        output.append(
            {
                "candidate_id": candidate_id,
                "filename": f"{candidate_id}.txt",
                "snippet": document.page_content[:200],
                "score": score,
            }
        )
    output.sort(key=lambda r: r["score"], reverse=True)
    return output
