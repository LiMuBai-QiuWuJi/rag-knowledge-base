import os
import json
import time 
from store import query_and_rerank
from skill.write_file.tool import write_file

script_dir = os.path.dirname(os.path.abspath(__file__))
eval_set_path = os.path.join(script_dir, "data", "eval_set.json")

with open(eval_set_path, 'r', encoding='utf-8') as file:
    eval_set = json.load(file)

results = []
hit_count = 0
i=1

print(f"开始评测")
start_time = time.time()    # 计时起点

for item in eval_set:
    start_time_item = time.time()
    question = item["question"]
    keywords = item["answer_keywords"]
    
    ranked = query_and_rerank(question_text=question, vector_top_k=10, rerank_top_n=5)
    documents = [r["text"] for r in ranked]
    full_text = "\n".join(documents)
    
    hit = any(kw in full_text for kw in keywords)
    if hit:
        hit_count += 1
    else:
        print(f"✗ 未命中：{question}")
    elapsed_item= time.time() - start_time_item
    print(f"第 {i} 块评测内容{question}，耗时{elapsed_item}秒")
    results.append({
        "question": question,
        "keywords": keywords,
        "documents": documents,
        "elapsed":elapsed_item,
        "hit": hit
    })
    i = i + 1

elapsed = time.time() - start_time   # 总耗时（秒）

accuracy = hit_count / len(results)
write_data = f"\n\n{'='*40}\n\n".join(
    json.dumps(r, ensure_ascii=False, indent=2) for r in results
)

write_file(
    txt_file_abs_path=os.path.join(os.path.dirname(eval_set_path), "评测结果输出文件.txt"),
    w_or_a="w",
    write_data=f"召回命中率：{accuracy:.2%}\n\n{write_data}\n总耗时：{elapsed:.1f} 秒\n\n"
)

print(f"\n评测完成：命中 {hit_count}/{len(results)}，命中率 {accuracy:.2%}，总耗时：{elapsed:.1f} 秒")
