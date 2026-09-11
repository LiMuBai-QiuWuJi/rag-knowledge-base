import os
import chromadb
from sentence_transformers import CrossEncoder

# 本地 rerank 模型路径（相对仓库根目录）
_RERANK_MODEL_PATH = "model/BAAI--bge-reranker-v2-m3/snapshots/master"
_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        if not os.path.isdir(_RERANK_MODEL_PATH):
            raise FileNotFoundError(f"未找到本地 rerank 模型目录：{_RERANK_MODEL_PATH}")
        _reranker = CrossEncoder(_RERANK_MODEL_PATH, device="cpu")
    return _reranker


def save_in_chroma(chunks: list[dict]):
    client = chromadb.PersistentClient(path=".chroma")
    collection = client.get_or_create_collection("my_kb")

    collection.add(
        ids=[f"{chunk['source']}_{chunk['chunk_index']}" for chunk in chunks],
        embeddings=[c["embedding"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "index": c["chunk_index"]} for c in chunks]
    )


def query_chroma(question_text: str = None, question_vector: list[float] = None, top_k: int = 5):
    """向量检索：支持传入向量或文本（文本会自动转向量）"""
    if question_text and not question_vector:
        from ingest import text_to_vector
        question_vector = text_to_vector(question_text)

    if not question_vector:
        raise ValueError("必须传入 question_vector 或 question_text")

    client = chromadb.PersistentClient(path=".chroma")
    collection = client.get_collection("my_kb")
    return collection.query(
        query_embeddings=[question_vector],
        n_results=top_k
    )


def delete_chroma(ids: str = None):
    if not ids or ids is None:
        raise ValueError("请传入非空 ids")

    client = chromadb.PersistentClient(path=".chroma")
    collection = client.get_collection("my_kb")
    collection.delete(ids=ids)


def rerank_documents(query: str, documents: list[str], top_n: int = None) -> list[dict]:
    """对候选文档做精排（rerank），返回按相关度降序排列的结果列表。
    每条结果包含：index（在原始列表中的位置）、score（分数）、text（文本）。
    """
    if not documents:
        return []

    reranker = _get_reranker()
    scores = reranker.predict([[query, doc] for doc in documents])
    indexed = [
        {"index": i, "score": float(scores[i]), "text": documents[i]}
        for i in range(len(documents))
    ]
    indexed.sort(key=lambda x: x["score"], reverse=True)
    if top_n:
        indexed = indexed[:top_n]
    return indexed


def query_and_rerank(question_text: str = None, question_vector: list[float] = None,
                     vector_top_k: int = 10, rerank_top_n: int = 5):
    """先向量召回 top_k，再 rerank 取 top_n，返回带分数的结果列表。
    注意：rerank 需要原始 query 文本，若仅传向量且 question_text 为空，则跳过 rerank。
    """
    results = query_chroma(
        question_text=question_text,
        question_vector=question_vector,
        top_k=vector_top_k,
    )
    
    documents = results.get("documents", [[]])[0]
    if not documents:
        return []

    if not question_text:
        return [{"index": i, "score": None, "text": doc} for i, doc in enumerate(documents)]

    return rerank_documents(question_text, documents, top_n=rerank_top_n)


# if __name__ == "__main__":
#     test_query = "1956年人工智能作为学科的诞生事件是什么？"
#     test_docs = [
#         "1950年，艾伦·图灵发表了开创性论文，提出了图灵测试。",
#         "1956年达特茅斯会议被认为是人工智能诞生的标志，约翰·麦卡锡等人首次提出人工智能一词。",
#         "1951年，图灵开发了第一个国际象棋程序。",
#         "1955年，逻辑理论家程序诞生，是第一个真正的AI程序。"
#     ]
#     ranked = rerank_documents(test_query, test_docs)
#     for item in ranked:
#         print(f"相关度分数: {item['score']:.4f}，文档: {item['text']}")


