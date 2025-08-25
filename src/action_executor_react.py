from state import AgentState, PlanStep
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from logger_config import logger
from settings import Settings
from agent_tools import get_all_tools
import time
import json
from langchain_openai import ChatOpenAI


def create_react_agent_instance(step: PlanStep) -> callable:
    """创建步骤专属ReAct执行器（与步骤元数据强绑定）"""
    # 1. 加载工具与LLM
    tools = get_all_tools()
    llm = ChatOpenAI(
        model=Settings.LLM_MODEL,
        temperature=Settings.TEMPERATURE,
        api_key=Settings.OPENAI_API_KEY,
        base_url=Settings.OPENAI_BASE_URL,
    )

    # 2. 构建步骤专属提示（适配ReAct格式）
    system_prompt = f"""你是步骤专属执行助手，仅完成以下单个步骤，严格遵循ReAct模式：

### 步骤信息
- 步骤ID：{step.id}
- 任务描述：{step.description}
- 指定工具：{step.tool if step.tool else "无（直接返回结果）"}
- 工具参数模板：{json.dumps(step.tool_args, ensure_ascii=False)}
- 输入模板：{step.input_template}
- 依赖步骤：{', '.join(step.dependencies) if step.dependencies else "无"}
- 预期输出：{step.expected_output}

### 可用工具
{{tools}}

### ReAct执行格式（必须严格遵守）
Question: 当前步骤需要完成的任务
Thought: 分析是否需要调用工具（依赖步骤结果是否足够）
Action: 工具名称（必须在{{tool_names}}中，无工具则跳过）
Action Input: 工具参数（需符合模板格式，可引用依赖步骤结果如 {{step_1_result}}）
Observation: 工具返回结果（无工具则直接写"无需工具"）
...（重复Thought-Action-Input-Observation）
Thought: 已完成当前步骤，获取预期输出
Final Answer: 步骤结果（JSON格式，便于后续步骤引用）

### 规则
1. 仅执行当前步骤，不处理其他任务
2. 工具参数必须是JSON对象（与模板结构一致）
3. 引用依赖步骤结果时用 {{step_id_result}} 格式
4. Final Answer 必须是结构化数据（优先JSON）
"""

    # 3. 创建ReAct执行器
    return create_react_agent(model=llm, tools=tools, prompt=system_prompt)


def run_single_step(step: PlanStep, state: AgentState) -> str:
    """执行单个步骤（调用ReAct执行器）"""
    logger.info(f"[执行节点] 执行步骤 | ID：{step.id} | 工具：{step.tool} | 描述：{step.description[:50]}...")

    try:
        # 1. 创建ReAct执行器
        agent = create_react_agent_instance(step)

        # 2. 准备输入消息（包含历史上下文，关键修复点：使用属性访问）
        # 替代原 state.get('context', {}) → 直接访问context属性
        history_messages = [
                               HumanMessage(
                                   content=f"基于以下上下文执行步骤{step.id}：\n{json.dumps(state.context, ensure_ascii=False)}")
                           ] + state.messages  # 直接访问messages属性

        # 3. 调用ReAct执行器
        result = agent.invoke({"messages": history_messages})

        # 4. 提取Final Answer（ReAct格式的最终结果）
        final_answer = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and "Final Answer:" in msg.content:
                final_answer = msg.content.split("Final Answer:")[-1].strip()
                break
        if not final_answer:
            # 降级：提取最后一条消息内容
            final_answer = str(result["messages"][-1].content)

        logger.info(f"[执行节点] 步骤成功 | ID：{step.id} | 结果长度：{len(final_answer)}字符")
        return final_answer

    except Exception as e:
        error_msg = f"步骤{step.id}执行失败：{str(e)}"
        logger.error(error_msg, exc_info=True)
        return json.dumps({"error": error_msg, "step_id": step.id})


def action_executor_node(state: AgentState) -> AgentState:
    """LangGraph执行节点：执行当前计划的第一个未完成步骤（适配纯类属性AgentState）"""
    current_plan = state.current_plan  # 直接访问current_plan属性
    if not current_plan:
        error_msg = "无有效计划可执行"
        state.add_message(AIMessage(content=f"❌ {error_msg}"))  # 使用封装方法添加消息
        state.need_replan = True  # 直接修改属性
        return state

    # 1. 筛选未执行步骤（按ID过滤，直接访问executed_steps属性）
    executed_ids = [s["step_id"] for s in state.executed_steps]
    unexecuted_steps = [s for s in current_plan.steps if s.id not in executed_ids]

    if not unexecuted_steps:
        # 2. 所有步骤已执行 → 标记任务完成（直接修改属性）
        state.task_completed = True
        state.need_replan = True  # 触发重规划确认完成
        state.add_message(AIMessage(content="✅ 所有步骤已执行完毕，等待任务确认"))
        logger.info("[执行节点] 所有步骤执行完成")
        return state

    # 3. 执行第一个未完成步骤
    current_step = unexecuted_steps[0]
    step_result = run_single_step(current_step, state)

    # 4. 记录已执行步骤（使用封装方法，关键修复点）
    state.add_executed_step(current_step, step_result)
    # （替代原 state["executed_steps"].append(...)）

    # 5. 更新消息与状态（使用属性访问和封装方法）
    state.add_message(AIMessage(
        content=f"📌 步骤{current_step.id}执行结果：\n描述：{current_step.description[:50]}...\n结果：{step_result[:100]}..."
    ))
    state.need_replan = True  # 执行后触发重规划
    # 记录错误（若有）
    state.last_error = "" if "error" not in step_result else step_result

    return state
