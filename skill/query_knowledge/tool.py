import sys
import os

# 将项目根目录加入 sys.path，确保能导入 ingest 和 store
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ingest import text_to_vector
from store import query_chroma


def query_knowledge(question: str, top_k: int = 5) -> str:
    """从本地知识库检索与问题相关的资料片段，返回格式化文本"""
    vec = text_to_vector(question)
    results = query_chroma(question_vector=vec, top_k=top_k)

    if not results["documents"] or not results["documents"][0]:
        return "知识库中没有找到相关资料。"

    chunks = []
    for i, (meta, doc) in enumerate(zip(results["metadatas"][0], results["documents"][0])):
        chunks.append(f"[{i + 1}] 来源：{meta['source']}（第{meta['index']}段）\n{doc}")

    return "\n\n".join(chunks)
