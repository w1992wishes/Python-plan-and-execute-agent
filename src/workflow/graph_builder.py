from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from src.workflow.state import AgentState, format_successful_tasks
# 导入所有异步节点
from src.workflow.intent_classifier_node import intent_classifier
from src.workflow.plan_node import plan
from src.workflow.execute_node import execute
from src.workflow.replan_node import replan
from langgraph.types import Send

def create_async_agent_workflow() -> CompiledStateGraph:
    """创建异步工作流（核心：指定异步节点）"""
    # 1. 初始化异步工作流
    workflow = StateGraph(AgentState)

    # 2. 添加异步节点（所有节点均为async def）
    workflow.add_node("classify_intent", intent_classifier)  # 异步意图分类
    workflow.add_node("plan", plan)                  # 异步规划
    workflow.add_node("execute", execute)            # 异步执行
    workflow.add_node("replan", replan)              # 异步重规划

    # 3. 定义异步流向（与同步逻辑一致）
    workflow.set_entry_point("classify_intent")
    workflow.add_edge("classify_intent", "plan")
    workflow.add_edge("plan", "execute")
    workflow.add_edge("execute", "replan")

    def plan_router(state: AgentState):
        current_step = state.get("current_plan").steps
        if not current_step and not current_step[0]:
            return "end"
        else:
            format_result_tasks = format_successful_tasks(state)
            step_tasks = current_step[0].step_tasks
            return [Send("execute", {"current_task": s, "format_successful_tasks": format_result_tasks}) for s in step_tasks]

    workflow.add_conditional_edges(
        "plan",
        plan_router,
        {"execute": "execute", "end": END}
    )

    def replan_router(state: AgentState):
        if state.get("task_completed"):
            return "end"
        else:
            current_step = state.get("current_plan").steps
            if not current_step and not current_step[0]:
                return "end"
            else:
                step_tasks = current_step[0].step_tasks
                return [Send("execute", {"current_task": s}) for s in step_tasks]

    workflow.add_conditional_edges(
        "replan",
        replan_router,
        {"execute": "execute", "end": END}
    )

    # 6. 编译异步工作流（关键：使用compile()，支持astream）
    return workflow.compile()