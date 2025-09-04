from dataclasses import dataclass
from typing import List


@dataclass
class StepTask:
    """
    并行任务模型（PlanStep 内可并行执行的子任务）
    特点：同 PlanStep 下的 StepTask 无依赖，可同时执行
    """
    id: str  # 任务唯一ID，格式：step_task_<步骤序号>_<任务序号>（如 step_task_1_2）
    description: str  # 任务描述（如："查询北京2024年Q1平均气温"）
    tool: str  # 关联工具名称（必须在 settings.ENABLED_TOOLS 中）
    confidence: float = 0.7  # 任务可行性置信度（0.0-1.0）


@dataclass
class PlanStep:
    """
    串行步骤模型（Plan 内按顺序执行，下一个依赖上一个的结果）
    新增：step_tasks 字段，存储当前步骤的并行子任务
    """
    id: str  # 步骤唯一ID，格式：step_1、step_2_123（序号+随机数）
    description: str  # 步骤描述（如："获取多城市气温数据并对比"）
    step_tasks: List[StepTask]  # 当前步骤的并行任务列表（核心新增字段）
    confidence: float = 0.7  # 步骤整体置信度（0.0-1.0）


@dataclass
class Plan:
    """
    完整任务计划模型（顶层结构，无修改，仅关联新增字段的 PlanStep）
    """
    id: str  # 计划唯一ID，格式：plan_1712345678（时间戳）
    query: str  # 用户原始查询（与AgentState.input一致）
    goal: str  # 计划总目标（如："对比北京/上海2024年Q1气温差异"）
    steps: List[PlanStep]  # 串行步骤列表（每个步骤含并行任务）
    confidence: float = 0.7  # 整体计划置信度（0.0-1.0）