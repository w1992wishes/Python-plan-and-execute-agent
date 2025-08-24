from state import AgentState
from logger_config import logger
from enum import Enum
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json
from settings import Settings
from json_util import extract_json_safely

class IntentType(str, Enum):
    """可复用的意图分类类型枚举（包含中英文表达）"""
    SIMPLE_QUERY = "指标简单查数"
    COMPARISON = "指标对比"
    ROOT_CAUSE_ANALYSIS = "指标根因分析"

    @classmethod
    def values(cls):
        """返回所有英文类型"""
        return [member.name for member in cls]

    @classmethod
    def chinese_labels(cls):
        """返回所有中文标签"""
        return [member.value for member in cls]


class IntentClassifierAgent:
    """问题意图分类Agent"""

    def __init__(self):
        """初始化分类器"""
        self.intent_types = list(IntentType)
        self.model = ChatOpenAI(
            model=Settings.LLM_MODEL,
            temperature=Settings.TEMPERATURE,
            api_key=Settings.OPENAI_API_KEY,
            base_url=Settings.OPENAI_BASE_URL,
        )

        # 构建提示词模板
        self.system_prompt = """你是一个专业的意图分类器，擅长分析用户问题的意图类型。
        请严格按照指定格式返回结果，不要添加任何额外解释。"""

        self.user_prompt_template = """
        请分析用户的问题，并将其分类为以下意图类型之一：
        {intent_types}

        请返回一个JSON格式的结果，包含以下字段：
        - intent_type: 选择的英文类型名称（必须是SIMPLE_QUERY、COMPARISON或ROOT_CAUSE_ANALYSIS之一）
        - chinese_label: 对应的中文标签
        - confidence: 分类的置信度（0-1之间的浮点数）

        用户问题：{question}
        """

    def classify(self, question: str) -> Dict[str, Any]:
        """
        对输入问题进行意图分类（使用LLM大模型）

        Args:
            question: 用户输入的问题文本

        Returns:
            包含分类结果的字典：
            - intent_type: 英文类型
            - chinese_label: 中文标签
            - confidence: 置信度
        """
        try:
            # 构建意图类型说明
            intent_type_descriptions = "\n".join(
                [f"- {intent.name}: {intent.value}" for intent in self.intent_types]
            )

            # 格式化提示词
            user_prompt = self.user_prompt_template.format(
                intent_types=intent_type_descriptions,
                question=question
            )

            # 准备消息
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_prompt)
            ]

            # 调用LLM模型
            response = self.model.invoke(messages)
            response_content = response.content.strip()

            # 解析LLM返回的结果
            result = extract_json_safely(response_content)

            # 验证结果格式
            required_fields = ["intent_type", "chinese_label", "confidence"]
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"LLM返回结果缺少必要字段: {field}")

            # 验证意图类型有效性
            if result["intent_type"] not in IntentType.values():
                raise ValueError(f"无效的意图类型: {result['intent_type']}")

            # 验证置信度范围
            if not (0 <= float(result["confidence"]) <= 1):
                raise ValueError(f"置信度必须在0-1之间: {result['confidence']}")

            # 记录分类结果
            logger.info(
                f"🎯 分类结果 | 问题：{question[:30]}... | 类型：{result['intent_type']}({result['chinese_label']}) | 置信度：{result['confidence']}"
            )

            return result

        except Exception as e:
            logger.error(f"意图分类出错: {str(e)}", exc_info=True)
            # 出错时返回默认分类
            return {
                "intent_type": IntentType.SIMPLE_QUERY.name,
                "chinese_label": IntentType.SIMPLE_QUERY.value,
                "confidence": 0.5
            }


def intent_classifier_node(state: AgentState):
    logger.info("🧠 意图问题分类节点已启动")
    """分类用户查询的意图"""
    user_query = state["input"]
    intent_agent = IntentClassifierAgent()
    return intent_agent.classify(question=user_query)
