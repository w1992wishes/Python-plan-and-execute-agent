# intent_classifier.py（适配新AgentState）
from state import AgentState  # 导入新AgentState
from logger_config import logger
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import Dict, Any

class IntentClassifierAgent:
    # 原有分类逻辑不变...
    def classify(self, question: str) -> Dict[str, Any]:
        try:
            # ... 调用LLM分类 ...
            return {
                "intent_type": "SIMPLE_QUERY",
                "chinese_label": "指标简单查数",
                "confidence": 0.9,
                "reason": "问题为简单查数场景"
            }
        except Exception as e:
            return {
                "intent_type": "SIMPLE_QUERY",
                "chinese_label": "指标简单查数",
                "confidence": 0.5,
                "reason": f"分类失败：{str(e)}"
            }


def intent_classifier_node(state: AgentState) -> AgentState:
    """现在state是纯类对象，只能用.state属性访问"""
    # 1. 访问input属性（绝对不会报AttributeError）
    logger.info("[意图分类节点] 启动 | 用户查询：%s", state.input[:50] + "...")

    try:
        # 2. 获取用户查询（直接访问属性）
        user_query = state.input
        if not user_query:
            raise ValueError("用户查询为空")

        # 3. 执行分类
        classifier = IntentClassifierAgent()
        classify_result = classifier.classify(question=user_query)

        # 4. 修改state属性（直接赋值或调用封装方法）
        state.set_intent_type(classify_result["intent_type"])  # 调用方法确保有效性
        state.intent_info = classify_result  # 直接赋值字典属性
        state.add_message(  # 调用封装方法添加消息
            AIMessage(
                content=f"🔍 意图分类完成：{classify_result['chinese_label']}（置信度：{classify_result['confidence']}）")
        )
        state.need_attention = "error" in classify_result  # 直接赋值布尔属性
        state.last_error = ""  # 清空错误记录

    except Exception as e:
        error_msg = f"意图分类失败：{str(e)}"
        logger.error(error_msg, exc_info=True)
        # 错误时修改state
        state.last_error = error_msg
        state.add_message(AIMessage(content=f"❌ {error_msg}"))
        state.need_attention = True

    # 5. 返回修改后的state对象（纯类对象，无字典混淆）
    return state