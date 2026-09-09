QUERY_KNOWLEDGE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_knowledge",
        "description": "从本地知识库中检索与用户问题相关的资料片段。当用户询问需要依据文档资料才能回答的问题时，调用此工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "用户的问题或查询关键词"
                },
                "top_k": {
                    "type": "integer",
                    "description": "召回的最相似片段数量，默认5",
                    "default": 5
                }
            },
            "required": ["question"]
        }
    }
}
