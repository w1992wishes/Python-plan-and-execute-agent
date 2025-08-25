from langgraph.graph import StateGraph, END
from state import AgentState
from task_planner import task_planner_node
from task_replanner import task_replanner_node
from action_executor_react import action_executor_node
from intent_classifier import intent_classifier_node
from langchain_core.messages import AIMessage
from logger_config import logger


def workflow_router(state: AgentState) -> str:
    """工作流路由：决定下一个节点"""
    if state.task_completed:
        return "end"  # 任务完成 → 结束
    elif state.need_replan:
        return "replan"  # 需要重规划 → 重规划节点
    elif not state.current_plan:
        return "plan"  # 无计划 → 规划节点
    else:
        return "execute"  # 有计划且无需重规划 → 执行节点


def create_agent_workflow() -> StateGraph:
    """创建完整的智能体工作流"""
    # 1. 初始化状态图（基于AgentState）
    workflow = StateGraph(AgentState)

    # 2. 添加核心节点
    workflow.add_node("classify_intent", intent_classifier_node)  # 意图分类
    workflow.add_node("plan", task_planner_node)  # 初始规划
    workflow.add_node("execute", action_executor_node)  # 步骤执行
    workflow.add_node("replan", task_replanner_node)  # 动态重规划

    # 3. 定义节点流向
    workflow.set_entry_point("classify_intent")  # 入口：意图分类
    workflow.add_edge("classify_intent", "plan")  # 意图分类 → 规划
    workflow.add_edge("plan", "execute")  # 规划 → 执行

    # 4. 条件路由：执行/重规划后动态决定流向
    workflow.add_conditional_edges(
        "execute",  # 从执行节点出发
        workflow_router,  # 路由逻辑
        {
            "replan": "replan",  # 需要重规划 → 重规划节点
            "execute": "execute",  # 继续执行 → 执行节点
            "end": END  # 任务完成 → 结束
        }
    )

    # 5. 重规划后路由
    workflow.add_conditional_edges(
        "replan",  # 从重规划节点出发
        workflow_router,
        {
            "execute": "execute",  # 重规划后执行 → 执行节点
            "end": END  # 任务完成 → 结束
        }
    )

    # 6. 编译工作流
    return workflow.compile()


# ------------------------------
# 工作流测试入口
# ------------------------------
if __name__ == "__main__":
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="智能体工作流测试")
    parser.add_argument("--query", type=str, default="2024年美国网球公开赛男单冠军是谁？",
                        help="测试用用户查询")
    args = parser.parse_args()

    # 1. 初始化工作流
    agent_graph = create_agent_workflow()

    # 2. 初始状态（与AgentState默认值对齐）
    initial_state = AgentState(
        input=args.query,
        messages=[],
        intent_type="SIMPLE_QUERY"
    )

    try:
        logger.info(f"[工作流测试] 开始处理查询：{args.query}")
        # 3. 执行工作流
        final_state = agent_graph.invoke(initial_state)

        # 4. 输出结果
        print("\n" + "=" * 50)
        print("📝 工作流执行结果")
        print("=" * 50)
        print(f"用户查询：{final_state.input}")
        print(f"任务状态：{'✅ 完成' if final_state.task_completed else '❌ 未完成'}")
        print(f"执行步骤数：{len(final_state.executed_steps)}")
        print(f"计划历史数：{len(final_state.plan_history)}")
        print("\n💬 最终回复：")
        for msg in reversed(final_state.messages):
            if hasattr(msg, "content") and "Final Answer" in msg.content or "任务执行完成" in msg.content:
                print(msg.content)
                break

    except Exception as e:
        logger.error(f"[工作流测试] 执行失败：{str(e)}", exc_info=True)