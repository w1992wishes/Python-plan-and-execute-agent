from langgraph.graph import StateGraph, END
from intent_classifier import intent_classifier_node
from task_planner import task_planner_node
from action_executor_react import action_executor_node
from response_evaluator import response_evaluator_node
from task_replanner import replan_node
from final_output import final_output_node
from state import AgentState


def build_agent_graph():
    """构建完整的 Agent 状态图"""
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("task_planner", task_planner_node)
    workflow.add_node("action_executor", action_executor_node)
    workflow.add_node("response_evaluator", response_evaluator_node)
    workflow.add_node("replan", replan_node)
    workflow.add_node("final_output", final_output_node)

    # 设置入口点
    workflow.set_entry_point("intent_classifier")

    # 添加普通边
    workflow.add_edge("intent_classifier", "task_planner")
    workflow.add_edge("task_planner", "action_executor")
    workflow.add_edge("action_executor", "response_evaluator")

    # 条件边：响应评估后决定重规划或最终输出
    workflow.add_conditional_edges(
        source="response_evaluator",
        path=lambda state: "replan" if state.get("need_replan", False) else "end",
        path_map={
            "replan": "replan",
            "end": "final_output"
        }
    )

    # 条件边：重规划后决定继续执行或终止（根据重规划限制）
    workflow.add_conditional_edges(
        source="replan",
        path=lambda state: "end" if state.get("replan_limit", False) else "action_executor",
        path_map={
            "action_executor": "action_executor",
            "end": "final_output"
        }
    )

    return workflow.compile()