from state import AgentState, Plan
from langchain_core.messages import AIMessage
from logger_config import logger
from prompt_setting import get_replanning_system_prompt
from plan_utils import BasePlanGenerator, validate_plan_for_react
import json


class TaskReplanner(BasePlanGenerator):
    """任务重规划器（适配ReAct执行器）"""

    def update_plan(self, state: AgentState) -> Plan:
        """根据执行状态更新计划（删除已完成步骤、修复参数）"""
        # 1. 提取核心状态信息（使用属性访问，关键修复点）
        original_plan = state.current_plan
        executed_steps = state.executed_steps
        error = state.last_error  # 直接访问last_error属性，替代state.get("last_error", "")
        query = state.input
        # 直接访问metadata字典的get方法（original_plan.metadata是字典）
        intent_type = original_plan.metadata.get("intent_type", "SIMPLE_QUERY")

        # 2. 格式化已执行步骤（供LLM参考结果）
        formatted_executed = self._format_executed_steps(executed_steps)
        # 格式化未执行步骤（供LLM调整）
        formatted_unexecuted = self._format_unexecuted_steps(original_plan, executed_steps)

        # 3. 构建重规划提示
        user_prompt = f"""### 重规划输入信息
- 用户查询：{query}
- 原计划ID：{original_plan.id}
- 已执行步骤（ReAct结果）：{formatted_executed}
- 未执行步骤：{formatted_unexecuted}
- 执行错误（若有）：{error}
- 可用工具：{self.tools_str}

### 重规划规则（必须遵守）
1. 删除已执行步骤（已执行ID：{[s['step_id'] for s in executed_steps]}）
2. 修复未执行步骤的参数格式（确保tool_args为JSON对象，适配ReAct）
3. 引用已执行步骤结果时用 {{step_id_result}} 格式（如 {{step_1_result}}）
4. 若所有步骤完成，返回空steps列表并在goal标记"任务完成"
5. 若执行错误是工具参数问题，优先修正参数格式
"""

        # 4. 调用LLM生成更新后计划
        system_prompt = get_replanning_system_prompt(intent_type=intent_type)
        updated_plan = self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            query=query
        )

        # 5. 校验更新后计划的兼容性
        valid, msg = validate_plan_for_react(updated_plan)
        if not valid:
            logger.warning(f"重规划计划校验失败：{msg}，重试生成")
            updated_plan = self.generate(
                system_prompt=system_prompt + "\n⚠️  计划需符合ReAct执行器格式（tool_args为JSON对象），请修正！",
                user_prompt=user_prompt,
                query=query
            )

        # 6. 补充重规划元数据
        updated_plan.metadata.update({
            "original_plan_id": original_plan.id,
            "updated_reason": error if error else "正常执行后更新",
            "executed_steps_count": len(executed_steps)
        })
        return updated_plan

    def _format_executed_steps(self, executed_steps: list) -> str:
        """格式化已执行步骤（突出ReAct结果）"""
        if not executed_steps:
            return "无已执行步骤"

        formatted = []
        for step in executed_steps:
            # 简化结果展示（避免过长）
            result = step["result"][:100] + "..." if len(str(step["result"])) > 100 else str(step["result"])
            formatted.append(f"步骤ID：{step['step_id']} | 工具：{step['tool_used']} | 结果：{result}")
        return "\n".join(formatted)

    def _format_unexecuted_steps(self, plan: Plan, executed_steps: list) -> str:
        """格式化未执行步骤（标记ReAct兼容性）"""
        executed_ids = [s["step_id"] for s in executed_steps]
        unexecuted = [s for s in plan.steps if s.id not in executed_ids]

        if not unexecuted:
            return "所有步骤已执行"

        formatted = []
        for step in unexecuted:
            # 标记参数是否兼容ReAct
            args_valid = "✅" if isinstance(step.tool_args, dict) else "❌"
            formatted.append(
                f"步骤ID：{step.id} | 工具：{step.tool} | 参数兼容：{args_valid} | 描述：{step.description[:50]}...")
        return "\n".join(formatted)


def task_replanner_node(state: AgentState) -> AgentState:
    """LangGraph重规划节点：更新计划并判断任务是否完成（适配纯类属性AgentState）"""
    logger.info(f"[重规划节点] 启动 | 原计划ID：{state.current_plan.id} | 已执行步骤：{len(state.executed_steps)}")

    try:
        # 1. 初始化重规划器并更新计划
        replanner = TaskReplanner()
        updated_plan = replanner.update_plan(state)

        # 2. 判断任务是否完成（空步骤 → 完成）
        if not updated_plan.steps:
            state.task_completed = True  # 直接修改属性，替代state["task_completed"] = True
            state.add_message(AIMessage(  # 使用封装方法添加消息
                content=f"🎉 任务执行完成！\n原计划ID：{state.current_plan.id}\n已执行步骤：{len(state.executed_steps)}\n最终结果：{state.executed_steps[-1]['result'][:150]}..."
            ))
            logger.info(
                f"[重规划节点] 任务完成 | 原计划ID：{state.current_plan.id} | 已执行步骤：{len(state.executed_steps)}")
            return state

        # 3. 任务未完成 → 更新计划状态（使用属性访问和封装方法）
        state.set_current_plan(updated_plan)  # 调用封装方法，替代state["current_plan"] = updated_plan
        state.plan_history.append(updated_plan)  # 直接访问列表属性，替代state["plan_history"].append(...)
        state.add_message(AIMessage(  # 使用封装方法添加消息
            content=f"🔄 重规划完成！\n新计划ID：{updated_plan.id}\n剩余步骤：{len(updated_plan.steps)}\n更新原因：{updated_plan.metadata['updated_reason'][:50]}..."
        ))
        state.need_replan = False  # 直接修改属性，替代state["need_replan"] = False
        state.last_error = ""  # 直接修改属性，替代state["last_error"] = ""

        logger.info(f"[重规划节点] 成功 | 新计划ID：{updated_plan.id} | 剩余步骤：{len(updated_plan.steps)}")
        return state

    except Exception as e:
        error_msg = f"重规划失败：{str(e)}"
        logger.error(error_msg, exc_info=True)

        # 4. 异常处理：保留原计划并标记重试
        state.add_message(AIMessage(content=f"❌ {error_msg}，将重试原计划"))  # 使用封装方法
        state.need_replan = True  # 直接修改属性，替代state["need_replan"] = True
        return state
