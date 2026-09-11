import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from store import save_in_chroma
from skill.read_file.tool import read_file

def chunk_text(text: str, source: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    chunks = []
    start = 0
    index = 0
    step = chunk_size - overlap  # 450
    
    while start < len(text):
        end = start + chunk_size
        chunks.append({
            "text": text[start:end],  # 最后一块自动取到结尾
            "source": source,
            "chunk_index": index
        })
        start += step
        index += 1
    
    return chunks

script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir,".env")
loadEnv = load_dotenv(dotenv_path=env_path)
print(f"Load \".env\" {loadEnv}")

def _get_embedding_client() -> OpenAI:
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        raise ValueError("获取 ZHIPU_API_KEY 失败。请检查.env文件或系统环境变量设置")
    return OpenAI(api_key=api_key, base_url="https://open.bigmodel.cn/api/paas/v4/")

def texts_to_vectors(texts: list[str], retries: int = 3, retries_time_s: int = 2) -> list[list[float]]:
    """输入文本列表，输出向量列表（带重试）"""
    client = _get_embedding_client()
    last_error = ""
    for attempt in range(retries):
        try:
            response = client.embeddings.create(model="embedding-2", input=texts)
            return [d.embedding for d in response.data]
        except Exception as e:
            print(f"第 {attempt} 次调用失败：{e}")
            last_error = e
            if attempt < retries - 1:
                time.sleep(retries_time_s)
    raise RuntimeError(f"{retries} 次重试失败，Error：{last_error}")

def text_to_vector(text: str, retries: int = 3, retries_time_s: int = 2) -> list[float]:
    """输入单个文本，输出单个向量（带重试）"""
    return texts_to_vectors([text], retries=retries, retries_time_s=retries_time_s)[0]

def do_vector(chunks: list[dict], retries: int = 3, retries_time_s: int = 2) -> list[dict]:
    """批量给 chunks 加 embedding 字段（带重试）"""
    texts = [c["text"] for c in chunks]
    vectors = texts_to_vectors(texts, retries=retries, retries_time_s=retries_time_s)
    for chunk, vec in zip(chunks, vectors):
        chunk["embedding"] = vec
    return chunks

def ingest(file_path: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """输入文件绝对地址，输出chunks 类型list[dict]。元素有 text, source, chunk_index, embedding"""
    # 延迟导入，避免循环导入
    text = read_file(file_path)
    chunks = chunk_text(text=text, source=file_path, chunk_size=chunk_size, overlap=overlap)
    chunks = do_vector(chunks=chunks)
    save_in_chroma(chunks=chunks)
    return chunks


# 测试用 
chunks = ingest(r"D:\\MyAgent\\code\\career-pivot\\rag-knowledge-base\data\\黄河水沙的通量分析(2).pdf")
# print(f"{chunks}")
chunks = ingest(r"D:\\MyAgent\\code\\career-pivot\\rag-knowledge-base\data\\阶段4-RAG与项目1知识库.docx")
# print(f"{chunks}")