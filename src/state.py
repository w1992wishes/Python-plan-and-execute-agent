from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import time


class PlanType(str, Enum):
    """计划类型枚举（顺序/并行）"""
    SEQUENTIAL = "sequential"  # 顺序执行（当前默认支持）
    PARALLEL = "parallel"      # 并行执行（预留扩展）


@dataclass
class PlanStep:
    """单个执行步骤模型（与ReAct执行器参数严格对齐）"""
    id: str  # 步骤唯一ID，格式：step_1、step_2_123（序号+随机数）
    description: str  # 步骤操作描述（如：调用tavily_search查询2024澳网冠军）
    tool: str = ""  # 关联工具名称（必须在settings.ENABLED_TOOLS中）
    tool_args: Dict[str, Any] = field(default_factory=dict)  # ReAct兼容的JSON参数
    input_template: str = ""  # 自然语言输入模板（如："查询{query}的{指标}"）
    dependencies: List[str] = field(default_factory=list)  # 依赖步骤ID列表
    expected_output: str = ""  # 预期输出描述（如："返回2024澳网男单冠军姓名"）
    confidence: float = 0.7  # 步骤可行性置信度（0.0-1.0）
    status: str = "pending"  # 步骤状态：pending（待执行）、completed（已完成）、failed（失败）


@dataclass
class Plan:
    """完整任务计划模型（关联多个步骤）"""
    id: str  # 计划唯一ID，格式：plan_1712345678（时间戳）
    query: str  # 用户原始查询（与AgentState.input一致）
    goal: str  # 计划总目标（如："获取2024澳网男单冠军及其家乡"）
    plan_type: PlanType  # 计划类型（当前仅支持SEQUENTIAL）
    steps: List[PlanStep]  # 步骤列表（按执行顺序排列）
    estimated_duration: float = 60.0  # 预计总耗时（秒）
    confidence: float = 0.7  # 整体计划置信度（0.0-1.0）
    metadata: Dict[str, Any] = field(
        default_factory=lambda: {
            "generated_by": "llm_planner",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "intent_type": "SIMPLE_QUERY"  # 关联意图类型
        }
    )
    created_at: float = field(default_factory=time.time)  # 生成时间戳（秒）
    updated_at: float = field(default_factory=time.time)  # 最后更新时间戳


@dataclass
class AgentState:
    """
    终极版Agent状态：纯类对象（非字典子类）
    所有属性通过实例.属性访问，彻底杜绝字典与类的混淆
    """
    # ------------------------------
    # 核心必填属性（带默认值，无需手动初始化）
    # ------------------------------
    input: str = ""  # 用户输入的原始查询（核心字段）
    intent_type: str = "SIMPLE_QUERY"  # 意图类型（英文，如SIMPLE_QUERY）
    intent_info: Dict[str, Any] = field(default_factory=dict)  # 完整意图结果（含中文标签、置信度）
    current_plan: Optional[Plan] = None  # 当前生效的计划（Plan对象或None）
    plan_history: List[Plan] = field(default_factory=list)  # 历史计划列表
    executed_steps: List[Dict[str, Any]] = field(default_factory=list)  # 已执行步骤记录
    need_replan: bool = False  # 是否需要重规划（布尔值）
    task_completed: bool = False  # 任务是否完成（布尔值）
    last_error: str = ""  # 上一次执行错误信息
    need_attention: bool = False  # 是否需要人工关注
    messages: List[Any] = field(default_factory=list)  # 交互消息列表（LangChain Message对象）
    context: Dict[str, Any] = field(default_factory=dict)  # 额外上下文（如用户历史对话）

    # ------------------------------
    # 属性操作方法（封装逻辑，避免直接修改）
    # ------------------------------
    def set_input(self, value: str) -> None:
        """设置用户查询（自动校验类型和去重）"""
        if not isinstance(value, str):
            raise TypeError(f"input必须是字符串类型，当前：{type(value).__name__}")
        self.input = value.strip()  # 自动去除前后空格

    def set_intent_type(self, intent_name: str) -> None:
        """设置意图类型（仅允许指定枚举值）"""
        valid_intents = ["SIMPLE_QUERY", "COMPARISON", "ROOT_CAUSE_ANALYSIS"]
        if intent_name not in valid_intents:
            raise ValueError(f"intent_type必须是{valid_intents}之一，当前：{intent_name}")
        self.intent_type = intent_name
        # 同步更新当前计划的意图类型（如果有计划）
        if self.current_plan:
            self.current_plan.metadata["intent_type"] = intent_name

    def add_executed_step(self, step: PlanStep, result: str) -> None:
        """添加已执行步骤记录（自动格式化，避免手动构造字典）"""
        if not isinstance(step, PlanStep):
            raise TypeError(f"step必须是PlanStep对象，当前：{type(step).__name__}")
        self.executed_steps.append({
            "step_id": step.id,
            "description": step.description,
            "tool_used": step.tool,
            "result": result,
            "status": "completed" if "error" not in str(result).lower() else "failed",
            "executed_at": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        # 执行后自动标记需要重规划
        self.need_replan = True

    def add_message(self, message: Any) -> None:
        """添加交互消息（确保是LangChain Message对象）"""
        from langchain_core.messages import BaseMessage
        if isinstance(message, BaseMessage) or hasattr(message, "content"):
            self.messages.append(message)
        else:
            raise TypeError(f"message必须是LangChain Message对象，当前：{type(message).__name__}")

    def set_current_plan(self, plan: Plan) -> None:
        """设置当前计划（自动更新历史和时间戳）"""
        if not isinstance(plan, Plan):
            raise TypeError(f"plan必须是Plan对象，当前：{type(plan).__name__}")
        # 添加到历史计划
        if self.current_plan:
            self.plan_history.append(self.current_plan)
        # 设置新计划
        self.current_plan = plan
        plan.updated_at = time.time()  # 更新计划时间戳

    # ------------------------------
    # 辅助方法
    # ------------------------------
    def _get_task_duration(self) -> float:
        """计算任务总耗时（从第一个计划生成到现在）"""
        if not self.plan_history and not self.current_plan:
            return 0.0
        first_plan = self.plan_history[0] if self.plan_history else self.current_plan
        return time.time() - first_plan.created_at

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化或日志）"""
        return {
            "input": self.input,
            "intent_type": self.intent_type,
            "intent_info": self.intent_info,
            "current_plan_id": self.current_plan.id if self.current_plan else None,
            "plan_history_count": len(self.plan_history),
            "executed_steps_count": len(self.executed_steps),
            "need_replan": self.need_replan,
            "task_completed": self.task_completed,
            "last_error": self.last_error,
            "message_count": len(self.messages),
            "task_duration": self._get_task_duration()
        }