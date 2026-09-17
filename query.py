import os
import sys
import json
from dotenv import load_dotenv
from call_llm import CallParameters, ChatSession, call_llm

from skill.calculator.tool import calculator
from skill.read_file.tool import read_file
from skill.write_file.tool import write_file
from skill.query_knowledge.tool import query_knowledge
from skill.calculator.schemas import CALCULATOR_SCHEMA
from skill.read_file.schemas import READFILE_SCHEMAS
from skill.write_file.schemas import WRITEFILE_SCHEMAS
from skill.query_knowledge.schemas import QUERY_KNOWLEDGE_SCHEMA

sys.stdout.reconfigure(encoding='utf-8')

script_dir = os.path.dirname(os.path.abspath(__file__))

env_path = os.path.join(script_dir, ".env")
load_dotenv(dotenv_path=env_path)

config_path = os.path.join(script_dir, "config.json")
with open(config_path, 'r', encoding='utf-8') as file:
    config_data = json.load(file)


def answer(question: str, session: ChatSession,
           stream: bool = True,
           context_mode: str = "unlimited",
           context_window: int = 10) -> str:
    """RAG 问答：LLM 自行决定是否需要调用 query_knowledge 工具检索资料"""
    tools = [CALCULATOR_SCHEMA, READFILE_SCHEMAS, WRITEFILE_SCHEMAS, QUERY_KNOWLEDGE_SCHEMA]

    api_key = os.getenv("DEEPSEEK_OPENAI_API_KEY")
    if not api_key:
        raise ValueError("获取 DEEPSEEK_API_KEY 失败。请检查.env文件或系统环境变量设置")

    params = CallParameters(
        api_key=api_key,
        base_url=config_data["base_url"],
        model=config_data["model"],
        max_tokens=config_data.get("max_tokens", 4096),
        temperature=config_data.get("temperature", 0),
        system_prompt="",
        user_input=question,
        stream=stream,
        tools=tools,
        tool_map={
            "calculator": calculator,
            "read_file": read_file,
            "write_file": write_file,
            "query_knowledge": query_knowledge,
        },
        context_mode=context_mode,
        context_window=context_window,
    )

    return call_llm(params, session=session)


def main():
    session = ChatSession(
        "你是一个助手。当用户的问题需要依据文档资料时，调用 query_knowledge 工具检索相关资料，"
        "然后基于检索结果回答，并标注引用来源。资料之外的知识不要使用。"
    )
    while True:
        user_input = input("你：")
        while not user_input:
            user_input = input("你：")
        if user_input.lower() in ("exit", "quit"):
            break
        elif user_input.lower() in ("showall",):
            print(f"\n\n{session.messages}")
            continue

        answer(user_input, session=session)
        print(f"\n{'=' * 40}\n")


if __name__ == "__main__":
    main()
