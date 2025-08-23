from state import AgentState
from logger_config import logger
from enum import Enum
from typing import Dict, Any
import random


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

    def classify(self, question: str):
        """
        对输入问题进行意图分类（当前为随机分类）

        Args:
            question: 用户输入的问题文本

        Returns:
            包含分类结果的字典：
            - intent_type: 英文类型（SIMPLE_QUERY/COMPARISON/ROOT_CAUSE_ANALYSIS）
            - chinese_label: 中文标签
            - confidence: 置信度
        """
        # 随机选择分类结果
        selected_type = random.choice(self.intent_types)

        # 生成随机置信度
        if selected_type == IntentType.SIMPLE_QUERY:
            confidence = round(random.uniform(0.7, 0.95), 2)
        elif selected_type == IntentType.COMPARISON:
            confidence = round(random.uniform(0.6, 0.85), 2)
        else:  # ROOT_CAUSE_ANALYSIS
            confidence = round(random.uniform(0.5, 0.8), 2)

        # 记录分类结果
        logger.info(
            f"🎯 分类结果 | 问题：{question[:30]}... | 类型：{selected_type.name}({selected_type.value}) | 置信度：{confidence}"
        )

        return {
            "intent_type": selected_type.name,  # 返回英文类型
            "chinese_label": selected_type.value,  # 返回中文标签
            "confidence": confidence
        }


def intent_classifier_node(state: AgentState):
    logger.info("🧠 意图问题分类节点已启动")
    """分类用户查询的意图"""
    user_query = state["input"]
    intent_agent = IntentClassifierAgent()
    return intent_agent.classify(question=user_query)