from typing import List, Any


def parse_messages(messages: List[Any]) -> None:
    """
    解析消息列表，打印 HumanMessage、AIMessage 和 ToolMessage 的详细信息

    Args:
        messages: 包含消息的列表，每个消息是一个对象
    """
    print("=== 消息解析结果 ===")
    for idx, msg in enumerate(messages, 1):
        print(f"\n消息 {idx}:")
        # 获取消息类型
        msg_type = msg.__class__.__name__
        print(f"消息类型: {msg_type}")

        # 提取消息内容
        content = getattr(msg, 'content', '<空>')
        print(f"内容: {content}")

        # 处理附加信息
        additional_kwargs = getattr(msg, 'additional_kwargs', {})
        if additional_kwargs:
            print("附加信息:")
            for key, value in additional_kwargs.items():
                if key == 'tool_calls' and value:
                    print("工具调用:")
                    for tool_call in value:
                        print(f" - ID: {tool_call['id']}")
                        print(f"   函数: {tool_call['function']['name']}")
                        print(f"   参数: {tool_call['function']['arguments']}")
                else:
                    print(f" {key}: {value}")

        # 处理 ToolMessage 特有字段
        if msg_type == 'ToolMessage':
            tool_name = getattr(msg, 'name', '')
            tool_call_id = getattr(msg, 'tool_call_id', '')
            print(f"工具名称: {tool_name}")
            print(f"工具调用 ID: {tool_call_id}")

        # 处理 AIMessage 的工具调用和元数据
        if msg_type == 'AIMessage':
            tool_calls = getattr(msg, 'tool_calls', [])
            if tool_calls:
                print("工具调用:")
                for tool_call in tool_calls:
                    print(f" - 名称: {tool_call['name']}")
                    print(f"   参数: {tool_call['args']}")
                    print(f"   ID: {tool_call['id']}")

            # 提取元数据
            metadata = getattr(msg, 'response_metadata', {})
            if metadata:
                print("元数据:")
                token_usage = metadata.get('token_usage', {})
                print(f" 令牌使用: {token_usage}")
                model_name = metadata.get('model_name', '未知')
                print(f" 模型名称: {model_name}")
                finish_reason = metadata.get('finish_reason', '未知')
                print(f" 完成原因: {finish_reason}")

        # 打印消息 ID
        msg_id = getattr(msg, 'id', '未知')
        print(f"消息 ID: {msg_id}")
        print("-" * 50)