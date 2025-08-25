from state import AgentState, Plan
from langchain_core.messages import AIMessage
from logger_config import logger
from prompt_setting import get_planning_system_prompt, create_planning_prompt
from plan_utils import BasePlanGenerator, validate_plan_for_react
import time


class TaskPlanGenerator(BasePlanGenerator):
    """初始计划生成器（适配ReAct执行器）"""

    def generate_initial_plan(self, state: AgentState) -> Plan:
        """根据Agent状态生成初始计划（整合意图、查询、上下文）"""
        # 1. 提取状态中的核心信息（使用纯属性访问）
        query = state.input
        intent_type = state.intent_type
        context = state.context  # 额外上下文（如用户历史对话）

        # 2. 获取相似历史计划（供LLM参考，提升计划质量）
        similar_plans = self._format_similar_plans(query)

        # 3. 构建LLM提示（调用统一提示词配置）
        user_prompt = create_planning_prompt(
            query=query,
            tools_str=self.tools_str,
            similar_plans_str=similar_plans,
            context=context
        )
        system_prompt = get_planning_system_prompt(intent_type=intent_type)

        # 4. 调用父类生成计划（复用LLM调用与解析逻辑）
        plan = self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            query=query
        )

        # 5. 二次校验计划兼容性（失败则重试一次）
        valid, msg = validate_plan_for_react(plan)
        if not valid:
            logger.warning(f"初始计划兼容性校验失败：{msg}，将重试生成")
            plan = self.generate(
                system_prompt=system_prompt + "\n⚠️  上一次计划格式不符合ReAct执行器要求，请严格按create_planning_prompt中的JSON格式生成！",
                user_prompt=user_prompt,
                query=query
            )

        # 6. 补充计划元数据（关联意图类型）
        plan.metadata["intent_type"] = intent_type
        return plan

    def _format_similar_plans(self, query: str) -> str:
        """格式化相似计划（突出ReAct兼容字段）"""
        similar_plans = super().get_similar_plans(query)
        if not similar_plans:
            return "无相似历史计划，需严格按格式生成新计划"

        formatted = []
        for idx, plan in enumerate(similar_plans[:2]):  # 最多展示2个
            # 仅保留符合ReAct格式的步骤
            valid_steps = [s for s in plan.steps if isinstance(s.tool_args, dict)]
            steps_info = ", ".join([f"步骤{s.id}（工具：{s.tool}）" for s in valid_steps[:2]])
            formatted.append(f"计划{idx + 1}：ID={plan.id[:15]}...，目标={plan.goal[:20]}...，有效步骤={steps_info}")
        return "; ".join(formatted)


def task_planner_node(state: AgentState) -> AgentState:
    """LangGraph规划节点：生成初始计划并更新状态（适配纯类属性AgentState）"""
    logger.info(f"[规划节点] 启动 | 用户查询：{state.input[:50]}... | 意图类型：{state.intent_type}")

    try:
        # 1. 生成初始计划
        generator = TaskPlanGenerator()
        initial_plan = generator.generate_initial_plan(state)

        # 2. 更新状态（使用属性访问而非字典赋值，关键修复点）
        state.set_current_plan(initial_plan)  # 调用封装方法设置当前计划
        # （替代原 state["current_plan"] = initial_plan）

        state.plan_history.append(initial_plan)  # 直接操作列表属性
        # （替代原 state["plan_history"].append(initial_plan)）

        state.add_message(AIMessage(  # 调用封装方法添加消息
            content=f"✅ 初始计划生成完成！\n计划ID：{initial_plan.id}\n目标：{initial_plan.goal}\n步骤数：{len(initial_plan.steps)}\n意图适配：{initial_plan.metadata['intent_type']}"
        ))
        # （替代原 state["messages"].append(...)）

        state.need_replan = False  # 直接赋值布尔属性
        state.task_completed = False  # 直接赋值布尔属性

        logger.info(f"[规划节点] 成功 | 计划ID：{initial_plan.id} | 步骤数：{len(initial_plan.steps)}")
        return state

    except Exception as e:
        error_msg = f"初始计划生成失败：{str(e)}"
        logger.error(error_msg, exc_info=True)

        # 3. 异常处理：更新错误状态（同样使用属性访问）
        state.add_message(AIMessage(content=f"❌ {error_msg}，已启用应急计划"))
        state.need_replan = True  # 直接赋值布尔属性
        state.last_error = error_msg  # 记录错误信息

        return state
