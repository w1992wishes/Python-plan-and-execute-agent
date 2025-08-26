from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from state import AgentState
# 导入所有异步节点
from intent_classifier import intent_classifier_node
from task_planner import task_planner_node
from action_executor_react import action_executor_node
from task_replanner import task_replanner_node

def create_async_agent_workflow() -> CompiledStateGraph:
    """创建异步工作流（核心：指定异步节点）"""
    # 1. 初始化异步工作流
    workflow = StateGraph(AgentState)

    # 2. 添加异步节点（所有节点均为async def）
    workflow.add_node("classify_intent", intent_classifier_node)  # 异步意图分类
    workflow.add_node("plan", task_planner_node)                  # 异步规划
    workflow.add_node("execute", action_executor_node)            # 异步执行
    workflow.add_node("replan", task_replanner_node)              # 异步重规划

    # 3. 定义异步流向（与同步逻辑一致）
    workflow.set_entry_point("classify_intent")
    workflow.add_edge("classify_intent", "plan")
    workflow.add_edge("plan", "execute")

    # 4. 条件路由（同步路由函数可直接用于异步工作流）
    def workflow_router(state: AgentState) -> str:
        if state.task_completed:
            return "end"
        elif state.need_replan:
            return "replan"
        else:
            return "execute"

    # 5. 添加条件边
    workflow.add_conditional_edges(
        "execute",
        workflow_router,
        {"replan": "replan", "execute": "execute", "end": END}
    )
    workflow.add_conditional_edges(
        "replan",
        workflow_router,
        {"execute": "execute", "end": END}
    )

    # 6. 编译异步工作流（关键：使用compile()，支持astream）
    return workflow.compile()