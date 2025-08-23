from typing import TypedDict, List, Literal, Optional, Dict, Any, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from enum import Enum
from dataclasses import dataclass


# ========== Plan 相关定义 ==========
class PlanType(Enum):
    """智能体支持的计划类型"""
    SEQUENTIAL = "sequential"  # 顺序执行步骤
    PARALLEL = "parallel"  # 并行执行（若支持）
    CONDITIONAL = "conditional"  # 条件执行
    ITERATIVE = "iterative"  # 迭代执行直到条件满足


@dataclass
class PlanStep:
    """计划中的单个步骤"""
    id: str  # 步骤唯一ID
    description: str  # 步骤描述
    tool: str  # 调用的工具名称
    tool_args: Dict[str, Any] # 工具入参
    input_template: str  # 输入模板（可选）
    dependencies: List[str]  # 依赖的步骤ID列表
    conditions: Optional[Dict[str, Any]] = None  # 执行条件（可选）
    expected_output: Optional[str] = None  # 预期输出描述（可选）
    confidence: float = 0.5  # 步骤置信度
    metadata: Dict[str, Any] = None  # 元数据（可选）

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Plan:
    """完整的执行计划"""
    id: str  # 计划唯一ID
    query: str  # 原始用户查询
    goal: str  # 计划目标
    plan_type: PlanType  # 计划类型（顺序/并行等）
    steps: List[PlanStep]  # 步骤列表
    estimated_duration: float = 0.0  # 预计执行时间（可选）
    confidence: float = 0.5  # 计划整体置信度
    metadata: Dict[str, Any] = None  # 元数据（可选）
    created_at: float = 0.0  # 创建时间戳（可选）

    def get_executable_steps(self, completed_steps: List[str]) -> List[PlanStep]:
        """获取当前可执行的步骤（满足依赖条件）"""
        executable = []
        for step in self.steps:
            if step.id not in completed_steps:
                # 检查所有依赖是否已完成
                if all(dep in completed_steps for dep in step.dependencies):
                    executable.append(step)
        return executable

    def is_complete(self, completed_steps: List[str]) -> bool:
        """判断计划是否已完成（所有步骤都执行完毕）"""
        return all(step.id in completed_steps for step in self.steps)


# ========== Agent 状态定义 ==========
class AgentState(TypedDict):
    """智能体运行时状态（通过 TypedDict 强约束结构）"""
    # 输入输出
    input: str
    output: Optional[str]
    context: Dict[str, Any] = None  # 上下文信息（可选）

    # 对话历史
    messages: Annotated[List[BaseMessage], operator.add]  # 消息列表（支持累加）

    # 意图分类结果
    intent_type: str  # 英文意图类型（如 SIMPLE_QUERY）
    chinese_label: str  # 中文意图标签（如 "指标简单查数"）

    # 计划相关
    current_plan: Optional[Plan]  # 当前执行的计划
    plan_history: Annotated[List[Plan], operator.add]  # 历史计划列表（支持累加）

    # 执行状态
    has_error: bool  # 是否发生错误
    error_message: Optional[str]   # 错误信息（可选）
    step_results: Dict[str, Any]  # 步骤执行结果（ID→结果）
    plan_failed: Optional[bool]  # 计划是否失败（可选）

    # 执行进度
    current_step: int = 0  # 当前执行到第几步
    max_steps: int = 10  # 最大步骤数

    # 重规划相关
    need_replan: Optional[bool]  # 是否需要重规划
    replan_count: int # 重规划次数
    replan_limit: Optional[bool]  # 重规划是否达上限
    replan_analysis: Dict[str, Any]  # 重规划分析结果（可选）


def create_initial_state(input_text: str, max_steps: int = 10) -> AgentState:
    """创建智能体初始状态"""
    return AgentState(
        input=input_text,
        output=None,
        current_step=0,
        max_steps=max_steps,
        has_error=False,
        error_message=None,
        step_results={},
        plan_failed=None,
        messages=[],
        intent_type="",
        chinese_label="",
        current_plan=None,
        plan_history=[],
        need_replan=False,
        replan_count=0,
        replan_limit=False,
        replan_analysis=None,
    )