import chromadb

def save_in_chroma(chunks: list[dict]):
    client = chromadb.PersistentClient(path=".chroma")
    collection = client.get_or_create_collection("my_kb")

    collection.add(
        ids=[f"{chunk['source']}_{chunk['chunk_index']}" for chunk in chunks],
        embeddings=[c["embedding"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "index": c["chunk_index"]} for c in chunks]
    )

def query_chroma(question_vector: list[float] = None, question_text: str = None, top_k: int = 5):
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

def delete_chroma(ids:str=None):
    if not ids or ids==None:
        raise ValueError("请传入非空ids")

    client = chromadb.PersistentClient(path=".chroma")
    collection = client.get_collection("my_kb")
    collection.delete(
        ids=ids
    )

#测试用
# result = query_chroma(question_text="embedding 与向量入库")
# print(result['documents'][0])  # 第一个问题的 top-k 文本列表

