import os
import sys
import json
import time
from dotenv import load_dotenv
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk
from openai.types.chat import ChatCompletion

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


class ChatSession:
    def __init__(self, content: str = ""):
        self.messages = []
        if content:
            self.messages.append({"role": "system", "content": content})

    def ask(self, content: str = ""):
        self.messages.append({"role": "user", "content": content})
        return self.messages

    def record_assistant(self, msg):
        if isinstance(msg, str):
            self.messages.append({"role": "assistant", "content": msg})
        else:
            msg_dict = msg.model_dump(exclude_none=True)
            self.messages.append(msg_dict)


def _dispatch_tool(func_name: str, args: dict) -> str:
    """根据函数名分发工具执行，返回结果字符串"""
    try:
        # 动态导入工具，避免 rag-knowledge-base 目录下无 skill/ 时报错
        if func_name == "calculator":
            return str(calculator(**args))
        elif func_name == "read_file":
            return str(read_file(**args))
        elif func_name == "write_file":
            return str(write_file(**args))
        elif func_name == "query_knowledge":
            return query_knowledge(**args)
        else:
            return f"Error: 未知工具 {func_name}"
    except ImportError:
        return f"Error: 工具 {func_name} 未配置（skill 目录不存在或路径不对）"
    except Exception as e:
        return f"Error: 工具执行异常 {e}"


def _run_tools(msg_dict: dict, session: ChatSession) -> None:
    """执行 assistant 消息里的全部 tool_calls，并把 role=tool 消息追加到 session"""
    for tc in msg_dict.get("tool_calls", []):
        func = tc.get("function", {})
        func_name = func.get("name", "")
        args_str = func.get("arguments", "{}")
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        result = _dispatch_tool(func_name, args)
        session.messages.append({
            "role": "tool",
            "tool_call_id": tc.get("id", ""),
            "content": str(result)
        })


def _extract_stream(response: Stream[ChatCompletionChunk]) -> tuple[dict, bool]:
    """
    消费流式 generator，拼接碎片化的 tool_calls。
    返回 (assistant_msg_dict, has_tool_calls)
    """
    assistant_content = ""
    accumulated_tool_calls = {}

    for chunk in response:
        delta = chunk.choices[0].delta

        if delta.content:
            assistant_content += delta.content
            print(delta.content, end="", flush=True)

        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in accumulated_tool_calls:
                    accumulated_tool_calls[idx] = {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": "",
                            "arguments": ""
                        }
                    }
                if tc.function:
                    if tc.function.name:
                        accumulated_tool_calls[idx]["function"]["name"] = tc.function.name
                    if tc.function.arguments:
                        accumulated_tool_calls[idx]["function"]["arguments"] += tc.function.arguments

        if chunk.choices[0].finish_reason is not None:
            break

    msg_dict = {
        "role": "assistant",
        "content": assistant_content,
    }
    has_tools = len(accumulated_tool_calls) > 0
    if has_tools:
        msg_dict["tool_calls"] = [accumulated_tool_calls[i] for i in sorted(accumulated_tool_calls.keys())]
    return msg_dict, has_tools


def _extract_nonstream(response: ChatCompletion) -> tuple[dict, bool]:
    """
    消费非流式 ChatCompletion。
    返回 (assistant_msg_dict, has_tool_calls)
    """
    msg = response.choices[0].message
    msg_dict = msg.model_dump(exclude_none=True)
    has_tools = msg.tool_calls is not None and len(msg.tool_calls) > 0
    if has_tools:
        print(f"DeepSeek: [调用工具 {len(msg.tool_calls)} 个]")
    else:
        print(f"DeepSeek: {msg_dict.get('content', '')}")
    return msg_dict, has_tools


def call_llm(session: ChatSession, config: dict, user_input: str = "",
             stream: bool = True, tools=None,
             retries: int = 3, retries_time_s: int = 2, timeout: float = 30.0) -> str:
    api_key = os.getenv("DEEPSEEK_OPENAI_API_KEY")
    if not api_key:
        raise ValueError("获取 DEEPSEEK_API_KEY 失败。请检查.env文件或系统环境变量设置")

    client = OpenAI(api_key=api_key, base_url=config["base_url"])

    last_error = ""
    session.ask(user_input)

    for attempt in range(retries):
        try:
            # 循环支持多轮工具调用，模型每轮请求工具都会回到这里重新 create
            while True:
                response = client.chat.completions.create(
                    model=config["model"],
                    messages=session.messages,
                    max_tokens=config["max_tokens"],
                    temperature=config["temperature"],
                    timeout=timeout,
                    stream=stream,
                    tools=tools,
                    tool_choice="auto"
                )

                if stream:
                    print("DeepSeek: ", end="", flush=True)
                    msg_dict, has_tools = _extract_stream(response)
                    print()
                    if has_tools:
                        print(f"调用工具 {len(msg_dict['tool_calls'])} 个")
                else:
                    msg_dict, has_tools = _extract_nonstream(response)

                session.messages.append(msg_dict)

                if not has_tools:
                    return msg_dict.get("content", "")

                _run_tools(msg_dict, session)
                # 继续循环，自动发起下一轮请求

        except Exception as e:
            print(f"第 {attempt} 次调用失败：{e}")
            last_error = e
            if attempt < retries - 1:
                time.sleep(retries_time_s)

    raise RuntimeError(f"{retries} 次重试失败，Error：{last_error}")


def answer(question: str,session:ChatSession) -> str:
    """RAG 问答：LLM 自行决定是否需要调用 query_knowledge 工具检索资料"""
    tools = [CALCULATOR_SCHEMA, READFILE_SCHEMAS, WRITEFILE_SCHEMAS, QUERY_KNOWLEDGE_SCHEMA]
    return call_llm(
        session=session,
        config=config_data,
        user_input=question,
        stream=True,
        tools=tools
    )


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
        elif user_input.lower() in ("showall"):
            print(f"\n\n{session.messages}")
            continue

        answer(user_input,session=session)
        print(f"\n{'=' * 40}\n")


if __name__ == "__main__":
    main()
