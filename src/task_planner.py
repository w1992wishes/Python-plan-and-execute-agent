from state import AgentState, Plan
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from logger_config import logger
from prompt_setting import get_planning_system_prompt, create_planning_prompt
from plan_utils import BasePlanGenerator


class TaskPlanGenerator(BasePlanGenerator):
    def __init__(self):
        super().__init__()
        # 替换为异步工具
        from agent_tools import get_all_tools
        self.tools = get_all_tools()
        self.tools_str = "\n".join([f"- {t.name}：{t.description}" for t in self.tools])

    async def generate_async(self, system_prompt: str, user_prompt: str, query: str) -> Plan:
        """异步计划生成入口（适配异步工作流，核心：LLM异步调用）"""
        try:
            # 1. 构建LLM输入消息
            messages = self._build_llm_messages(system_prompt, user_prompt)
            logger.debug(f"[异步计划生成器] 调用LLM，消息数：{len(messages)}，查询预览：{query[:30]}...")

            # 2. 异步调用LLM（关键：替换invoke为ainvoke）
            response = await self.llm.ainvoke(messages)
            if not hasattr(response, "content") or not response.content:
                raise ValueError("LLM返回空内容，无法生成计划")

            # 3. 解析LLM响应并生成Plan对象（复用同步解析逻辑）
            return self._parse_llm_response(query, response.content)

        except Exception as e:
            error_msg = f"异步计划生成失败：{str(e)}"
            logger.error(error_msg, exc_info=True)
            # 降级：返回应急计划
            return self._parse_llm_response(query, f'{{"error": "{error_msg}", "steps": []}}')

    async def generate_initial_plan(self, state: AgentState) -> Plan:
        """异步生成初始计划"""
        query = state.input
        intent_type = state.intent_type
        user_prompt = create_planning_prompt(query=query, tools_str=self.tools_str)
        system_prompt = get_planning_system_prompt(intent_type=intent_type)
        # 异步调用生成计划
        plan = await self.generate_async(system_prompt, user_prompt, query)  # 关键：await
        plan.metadata["intent_type"] = intent_type
        return plan

# 异步规划节点
async def task_planner_node(state: AgentState) -> AgentState:
    logger.info(f"[规划节点] 启动 | 查询：{state.input[:50]}... | 意图：{state.intent_type}")
    generator = TaskPlanGenerator()
    initial_plan = await generator.generate_initial_plan(state)  # 异步调用
    # 状态更新（纯类属性访问）
    state.set_current_plan(initial_plan)
    state.add_message(AIMessage(
        content=f"✅ 初始计划生成完成！\n计划ID：{initial_plan.id}\n步骤数：{len(initial_plan.steps)}"
    ))
    logger.info(f"[规划节点] 完成 | 步骤数：{len(initial_plan.steps)}")
    return state

