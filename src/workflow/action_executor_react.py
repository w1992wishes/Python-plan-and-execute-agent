from src.workflow.state import AgentState
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent
from src.log.logger import logger
from src.setting.settings import Settings
from src.workflow.agent_tools import get_all_tools
from langchain_openai import ChatOpenAI

def create_execute_agent() -> callable:
    """创建异步ReAct Agent（使用异步工具）"""
    tools = get_all_tools()  # 加载异步工具
    llm = ChatOpenAI(
        model=Settings.LLM_MODEL,
        temperature=0.2,
        api_key=Settings.OPENAI_API_KEY,
        base_url=Settings.OPENAI_BASE_URL,
    )
    # 构建步骤专属提示（保持原有）
    system_prompt = """You are a helpful assistant that can use tools to answer questions.
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original question

    Begin!"""
    # 创建异步Agent（langgraph.prebuilt支持异步）
    return create_react_agent(model=llm, tools=tools, prompt=system_prompt)

# 异步执行节点
async def action_executor_node(state: AgentState) -> AgentState:
    current_plan = state.current_plan
    if not current_plan:
        error_msg = "无有效计划可执行"
        state.add_message(AIMessage(content=f"❌ {error_msg}"))
        state.need_replan = True
        return state

    # 提取第一个步骤的描述（plan 是 Step 列表，需通过 .description 获取任务内容）
    current_step = current_plan.steps[0]
    plan_str = "\n".join(f"{step.id}. {step.description}" for step in current_plan.steps)

    # 构造 agent 任务（明确要执行的步骤）
    task_formatted = f"""
        你的任务是执行以下计划的第 {current_step.id} 步：
        完整计划：
        {plan_str}

        当前需执行的步骤：{current_step.description}
        请执行该步骤（例如：调用工具查询信息），并返回执行结果（无需格式，直接文字描述）。
        """

    # 调用 agent 执行步骤（确保 agent_executor 接受 {"messages": [...]}）
    step_result = await create_execute_agent().ainvoke(
        {"messages": [("user", task_formatted)]}
    )

    # 更新状态（使用封装方法）
    result = step_result["messages"][-1].content
    state.add_executed_step(current_step, result)
    state.add_message(AIMessage(
        content=f"📌 步骤{current_step.id}执行结果：{result}..."
    ))
    logger.info(f"[执行节点] 步骤完成 | ID：{current_step.id}")
    return state