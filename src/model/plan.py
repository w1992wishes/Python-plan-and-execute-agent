from dataclasses import dataclass
from typing import List

@dataclass
class PlanStep:
    """单个执行步骤模型（与ReAct执行器参数严格对齐）"""
    id: str  # 步骤唯一ID，格式：step_1、step_2_123（序号+随机数）
    description: str  # 步骤操作描述（如：调用tavily_search查询2024澳网冠军）
    tool: str = ""  # 关联工具名称（必须在settings.ENABLED_TOOLS中）
    confidence: float = 0.7  # 步骤可行性置信度（0.0-1.0）


@dataclass
class Plan:
    """完整任务计划模型（关联多个步骤）"""
    id: str  # 计划唯一ID，格式：plan_1712345678（时间戳）
    query: str  # 用户原始查询（与AgentState.input一致）
    goal: str  # 计划总目标（如："获取2024澳网男单冠军及其家乡"）
    steps: List[PlanStep]  # 步骤列表（按执行顺序排列）
    confidence: float = 0.7  # 整体计划置信度（0.0-1.0）