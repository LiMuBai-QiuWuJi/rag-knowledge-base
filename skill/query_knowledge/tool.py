import sys
import os

# 将项目根目录加入 sys.path，确保能导入 ingest 和 store
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ingest import text_to_vector
from store import query_and_rerank


def query_knowledge(question: str, top_k: int = 5) -> str:
    """从本地知识库检索与问题相关的资料片段，先向量召回再 rerank 精排，返回格式化文本"""
    ranked = query_and_rerank(
        question_text=question,
        vector_top_k=10,
        rerank_top_n=top_k
    )

    if not ranked:
        return "知识库中没有找到相关资料。"

    chunks = []
    for i, item in enumerate(ranked):
        score_text = f" 相关度：{item['score']:.4f}" if item['score'] is not None else ""
        chunks.append(f"[{i + 1}]{score_text}\n{item['text']}")

    return "\n\n".join(chunks)
