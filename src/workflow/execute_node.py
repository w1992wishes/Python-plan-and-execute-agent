from src.workflow.state import AgentState, format_successful_tasks
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent
from src.log.logger import logger
from src.setting.settings import Settings
from src.workflow.agent_tools import get_all_tools
from langchain_openai import ChatOpenAI
from src.utils.json_util import extract_json_safely

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
async def execute(state: AgentState):
    current_task = state.get("current_task", None)
    if not current_task:
        error_msg = "无有效任务可执行"
        return {
            "need_replan": True,
            "message": [AIMessage(content=f"❌ {error_msg}")],
        }

    logger.info(
        f"[执行节点] 开始执行任务：{current_task.description} | {current_task.tool}"
    )

    task_results = state.get("format_successful_tasks", "")
    # 构造 agent 任务（明确要执行的步骤）
    task_formatted = f"""
        你的任务是调用工具执行以下任务：
        当前需执行的步骤：{current_task.description}
        已有的信息：{task_results}
        请执行该步骤（例如：调用工具查询信息），并返回执行结果，用json输出。
        
        {{
            "result": "工具执行结果（如查询到的数值、计算结果等）"
            "status": "success/failed",  # 执行状态
            "error_message": "若失败，简要说明失败原因（如无则为空）"
        }}
        """

    # 调用 agent 执行步骤（确保 agent_executor 接受 {"messages": [...]}）
    step_result = await create_execute_agent().ainvoke(
        {"messages": [("user", task_formatted)]}
    )

    # 更新状态（使用封装方法）
    result = extract_json_safely(step_result["messages"][-1].content)
    logger.info(f"[执行节点] 步骤完成 | 任务 {current_task.description} ：结果： {str(result)}")
    return {
        "messages": [AIMessage(content=f"✅ 步骤执行完成！\n任务：{current_task.description}\n结果预览：{str(result)[:100]}...")],
        "executed_tasks": [{
            "description": current_task.description,
            "tool_used": current_task.tool,
            "result": result.get("result", ""),
            "status": result.get("status", "failed"),
            "error_message": result.get("error_message", "")
        }]
    }