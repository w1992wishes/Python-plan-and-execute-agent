"""
final_output.py
最终输出节点：生成自然语言响应
"""

from state import AgentState
from langchain_core.messages import AIMessage
from logger_config import logger


def final_output_node(state: AgentState) -> dict:
    """生成最终响应消息"""
    logger.info("🚀📢 [任务报告节点启动，正在整理输出...]")
    if not state["output"]:
        content = "抱歉，我无法获取足够的信息来回答您的问题。"
    else:
        content = "查询结果：\n" + state["output"]
    return {"messages": [AIMessage(content=content)]}