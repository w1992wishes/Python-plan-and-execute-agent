from state import AgentState
from logger_config import logger
from langchain_core.messages import AIMessage
from task_planner import Plan  # 假设 task_planner 定义了 Plan 类
import json


def _analyze_execution_results(
        current_plan: Plan,
        executed_steps: list,
        step_results: dict
) -> dict:
    """
    分析已执行步骤的结果，判断是否需要重新规划

    返回格式：
    {
        "need_replan": bool,
        "reason": str,
        "issues": list,
        "suggested_adjustments": list
    }
    """
    analysis = {
        "need_replan": False,
        "reason": "无需重新规划",
        "issues": [],
        "suggested_adjustments": []
    }

    # 检查是否有步骤执行失败
    failed_steps = []
    for step_id, result in step_results.items():
        if isinstance(result, Exception) or (isinstance(result, dict) and result.get("error")):
            failed_steps.append(step_id)
            analysis["issues"].append(f"步骤 {step_id} 执行失败: {str(result)}")

    if failed_steps:
        analysis["need_replan"] = True
        analysis["reason"] = f"发现 {len(failed_steps)} 个步骤执行失败"
        analysis["suggested_adjustments"].append("跳过失败步骤或尝试替代方案")

    # 检查结果是否符合预期
    unexpected_results = []
    for step in executed_steps:
        result = step_results.get(step)
        if not result or (isinstance(result, dict) and not result.get("result")):
            continue

        # 检查结果中是否包含异常关键词
        if "异常" in str(result) or "错误" in str(result) or "失败" in str(result):
            unexpected_results.append(step)
            analysis["issues"].append(f"步骤 {step} 结果不符合预期: {str(result)}")

    if unexpected_results:
        analysis["need_replan"] = True
        analysis["reason"] = f"发现 {len(unexpected_results)} 个步骤结果异常"
        analysis["suggested_adjustments"].append("调整后续步骤以处理异常结果")

    # 检查是否有新的风险需要考虑（针对 risk_analysis 工具结果）
    risk_steps = [
        step_result
        for step_result in step_results.values()
        if isinstance(step_result, dict) and step_result.get("tool") == "risk_analysis"
    ]

    for step in risk_steps:
        try:
            result = step.get("result", {})
            if not isinstance(result, dict):
                logger.warning(f"无效的风险结果格式: {type(result)}")
                continue

            # 检查风险标志
            if result.get("has_risk"):
                analysis.setdefault("issues", [])
                analysis.setdefault("suggested_adjustments", [])

                analysis["need_replan"] = True
                analysis["reason"] = "检测到业务风险，需要调整计划"

                # 获取风险详情（带默认值）
                risk_level = result.get("risk_level", "medium")
                risk_details = result.get("risk_details", ["风险详情未提供"])

                # 添加风险条目
                analysis["issues"].append(
                    f"检测到 {risk_level} 级风险: {', '.join(risk_details)}"
                )

                # 添加调整建议（避免重复）
                if "增加风险缓解步骤" not in analysis["suggested_adjustments"]:
                    analysis["suggested_adjustments"].append("增加风险缓解步骤")

        except Exception as e:
            logger.error(f"处理风险步骤时出错: {str(e)}", exc_info=True)

    return analysis


def response_evaluator_node(state: AgentState) -> dict:
    """响应评估节点：判断执行结果是否需要重规划"""
    logger.info("🔍 评估执行结果节点启动，评估执行结果中...")

    current_plan = state.get("current_plan")
    step_results = state.get("step_results", {})
    executed_steps = list(step_results.keys())
    result = state.get("output")

    # 无结果时直接标记重规划
    if not result:
        logger.warning("没有生成结果")
        return {
            "need_replan": True,
            "messages": state.get("messages", []) + [
                AIMessage(content="没有生成结果，需要重新规划任务生成")
            ]
        }

    # 初步假设无需重规划（实际需结合 _analyze_execution_results 结果，此处示例为完整逻辑）
    replan_analysis = _analyze_execution_results(
        current_plan=current_plan,
        executed_steps=executed_steps,
        step_results=step_results
    )

    # 根据分析结果构造返回
    return {
        "need_replan": replan_analysis["need_replan"],
        "messages": state.get("messages", []) + [
            AIMessage(content=replan_analysis["reason"])
        ],
        "issues": replan_analysis["issues"],
        "suggested_adjustments": replan_analysis["suggested_adjustments"]
    }