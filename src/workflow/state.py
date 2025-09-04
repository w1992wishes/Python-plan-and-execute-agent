from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Annotated, TypedDict
from src.model.plan import Plan, StepTask
import operator

@dataclass
class AgentState(TypedDict):
    """
    终极版Agent状态：纯类对象（非字典子类）
    所有属性通过实例.属性访问，彻底杜绝字典与类的混淆
    """
    messages: Annotated[List[Any], operator.add]

    plan_history: Annotated[List[Plan], operator.add]

    executed_tasks: Annotated[List[Dict[str, Any]], operator.add]  # 已执行步骤记录
    current_task: StepTask

    input: str  # 用户输入的原始查询（核心字段）
    intent_type: str  # 意图类型（英文，如SIMPLE_QUERY）
    intent_info: Dict[str, Any]  # 完整意图结果（含中文标签、置信度）

    current_plan: Optional[Plan]  # 当前生效的计划（Plan对象或None）

    need_replan: bool  # 是否需要重规划（布尔值）
    task_completed: bool  # 任务是否完成（布尔值）
    last_error: str  # 上一次执行错误信息
    need_attention: bool   # 是否需要人工关注

def format_successful_tasks(state) -> str:
    executed_tasks = state.get("executed_tasks", [])
    seen_task_names = set()
    unique_successful = []

    for task in executed_tasks:
        if task.get("status") == "success":
            task_name = task.get("description", "未知任务")
            if task_name not in seen_task_names:
                seen_task_names.add(task_name)
                unique_successful.append({
                    "name": task_name,
                    "result": task.get("result", "无结果")
                })

    # 2. 格式化为"任务名，任务结果"的字符串（每行一个）
    return "\n".join([
        f"{task['name']}，{task['result']}"
        for task in unique_successful
    ])