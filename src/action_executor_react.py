from agent_tools import get_all_tools
from logger_config import logger
from langchain_openai import ChatOpenAI
from settings import Settings
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, ToolMessage
from typing import TypedDict, Annotated, List, Sequence
import operator
import json
from langchain.tools.render import render_text_description
from langgraph.graph import StateGraph, END  # 图构建依赖
from pprint import pprint

class ReActAgentState(TypedDict):
    """定义 Agent 在图中的状态，所有节点共享和修改此状态。"""
    messages: Annotated[Sequence[BaseMessage], operator.add]


class ReActAgent:
    def __init__(self, model: ChatOpenAI, tools: List, system_message: str | None = None):
        """
        初始化 Agent。
        - model: 绑定工具的 LangChain Chat Model 实例
        - tools: LangChain 工具实例列表
        """
        self.model = model
        self.tools = tools
        self.tools_map = {t.name: t for t in tools}  # 工具名称映射
        self.graph = self._build_graph()
        self.conversation_history = []  # 对话历史存储
        self.system_message = SystemMessage(content=system_message) if system_message else None

    def _build_graph(self) -> StateGraph:
        """构建并编译 LangGraph 图。"""
        workflow = StateGraph(ReActAgentState)

        # 添加节点
        workflow.add_node("agent_llm", self._call_model)
        workflow.add_node("action", self._call_tool)

        # 设置入口点
        workflow.set_entry_point("agent_llm")

        # 添加条件边
        workflow.add_conditional_edges(
            "agent_llm",
            self._should_continue,
            {
                "continue": "action",
                "end": END,
            },
        )

        # 添加普通边
        workflow.add_edge("action", "agent_llm")

        # 编译图
        return workflow.compile()

    def _call_model(self, state: ReActAgentState) -> dict:
        """
        私有方法：调用大模型。
        这是图中的 "agent_llm" 节点。
        """
        messages = state['messages']
        print(f"llm message: {messages}")
        model_with_tools = self.model.bind_tools(self.tools)
        response = model_with_tools.invoke(messages)
        # 统计 token 使用情况
        return {"messages": [response]}

    def _call_tool(self, state: ReActAgentState) -> dict:
        """
        私有方法：调用工具。
        这是图中的 "action" 节点。
        """
        last_message = state['messages'][-1]
        print(f"tool message: {last_message}")

        if not last_message.tool_calls:
            return {}

        tool_messages = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            if tool_name in self.tools_map:
                tool_to_call = self.tools_map[tool_name]
                try:
                    # 调用工具并获取输出
                    tool_output = tool_to_call.invoke(tool_call['args'])
                    # 将结构化输出序列化为字符串
                    tool_output_str = json.dumps(tool_output, ensure_ascii=False)

                    tool_messages.append(
                        ToolMessage(
                            content=tool_output_str,
                            tool_call_id=tool_call['id'],
                        )
                    )
                except Exception as e:
                    error_msg = f"Error executing tool {tool_name}: {e}"
                    tool_messages.append(
                        ToolMessage(content=error_msg, tool_call_id=tool_call['id'])
                    )
            else:
                # 如果模型尝试调用一个不存在的工具
                error_msg = f"Tool '{tool_name}' not found."
                tool_messages.append(
                    ToolMessage(content=error_msg, tool_call_id=tool_call['id'])
                )

        return {"messages": tool_messages}

    def _should_continue(self, state: ReActAgentState) -> str:
        """
        私有方法：决策下一步走向。
        这是图中的条件边逻辑。
        """
        last_message = state['messages'][-1]
        if last_message.tool_calls:
            return "continue"
        else:
            return "end"

    def run(self, query: str, stream: bool = True) -> str:
        """
        运行 Agent 处理单个查询。
        - query: 用户的输入问题。
        - stream: 是否流式打印中间步骤 (默认为 True)。
        返回 Agent 的最终回答。
        """
        initial_messages = [self.system_message] if self.system_message else []
        current_messages = initial_messages + self.conversation_history + [HumanMessage(content=query)]
        inputs = {"messages": current_messages}

        final_answer = ""  # 初始化一个变量来存储最终答案

        if stream:
            print(f"--- Running query: {query} ---\n")

            # --- 只运行一次图 ---
            for output in self.graph.stream(inputs):
                # 1. 打印日志（保持不变）
                print("--- Node Output ---")
                pprint(output)
                print("\n")

                # 2. 智能捕获最终答案
                # 最终答案的特征是：它来自于'agent_llm'节点，并且不包含工具调用
                for key, value in output.items():
                    if key == 'agent_llm':
                        # 检查 'agent_llm' 节点的输出中是否有消息
                        messages = value.get('messages', [])
                        if messages:
                            last_message = messages[-1]
                            # 如果最后一条消息不是工具调用，那么它就是最终答案
                            if not last_message.tool_calls:
                                final_answer = last_message.content
            if final_answer:
                # 1. 获取当前用户的提问
                user_message = HumanMessage(content=query)
                # 2. 将最终答案包装成 AIMessage
                agent_message = AIMessage(content=final_answer)
                # 3. 将这一对 Q&A 追加到长期历史中
                self.conversation_history.extend([user_message, agent_message])
            return final_answer

        else:
            # 非流式模式保持不变，因为它只运行一次 invoke
            final_state = self.graph.invoke(inputs)
            user_message = HumanMessage(content=query)
            agent_message = AIMessage(content=final_state['messages'][-1].content)
            self.conversation_history.extend([user_message, agent_message])
            return final_state['messages'][-1].content


def create_agent(plan_steps: str):
    """创建 ReAct 智能体（注入任务清单和工具）"""
    tools = get_all_tools()
    rendered_tools = render_text_description(tools)

    # 构造系统提示（含任务流程、工具、格式约束）
    system_prompt = f"""你是非常靠谱的任务执行助手，请严格遵守任务清单，按照顺序调用工具进行执行：

可用工具列表：
\n{rendered_tools}\n

任务清单：
\n{plan_steps}\n

请严格按照以下规则执行：
1. 严格按照任务清单顺序执行，不能跳过或更改顺序
2. 每次只能调用一个工具，等待工具结果返回后再继续
3. 工具调用后，必须反思工具结果，判断是否需要将从任务结果中取出数据放到下一步的任务参数中，不能盲目执行
4. 如果任务清单中某步没有指定工具或某步指定的工具不存在，则思考后跳过该步骤，继续执行下一步
5. 如果任务清单中所有步骤都执行完毕，则思考后给出最终答案
""".strip()

    # 初始化大模型
    llm = ChatOpenAI(
        model=Settings.LLM_MODEL,
        temperature=Settings.TEMPERATURE,
        api_key=Settings.OPENAI_API_KEY,
        base_url=Settings.OPENAI_BASE_URL,
    )
    return ReActAgent(model=llm, tools=tools, system_message=system_prompt)


from state import AgentState
from langchain_core.messages import AIMessage
import traceback


# ... 其他已有导入 ...


def action_executor_node(state: AgentState) -> dict:
    """LangGraph 节点：执行 ReAct 智能体，含异常处理与重规划标记"""
    logger.info("🚀🤖 ReAct 智能体启动，开始执行任务")
    current_plan = state.get("current_plan")

    # 无任务计划的快速失败
    if not current_plan:
        logger.warning("无有效执行计划，直接返回失败")
        return {"messages": [AIMessage(content="无法执行：当前无任务计划")]}

    # 构建带状态和依赖的任务清单
    plan_steps_text = "任务清单：\n"
    for i, step in enumerate(current_plan.steps):
        status = "⏳"
        plan_steps_text += f"{i + 1}. [{status}] {step.description}"

        # 附加工具信息
        if step.tool:
            plan_steps_text += f"（工具：{step.tool}）"
        plan_steps_text += "\n"

        # 依赖说明（非首步添加依赖提示）
        if i > 0:
            plan_steps_text += f"  依赖：步骤 {i} 的结果可能影响当前步骤参数\n"

    try:
        # 创建智能体并执行任务
        agent = create_agent(plan_steps_text)
        input_text = (
            f"用户原始查询：{state['input']}\n"
            f"执行目标：{current_plan.goal}\n"
            "请严格按照任务清单顺序执行，任务完成后反思结果并动态调整参数。"
        )
        logger.info(f"📥 智能体输入：\n{input_text}")
        logger.info(f"📋 任务清单：\n{plan_steps_text}")

        result = agent.run(input_text)
        # 组装返回消息（含执行结果）
        messages = state.get("messages", []) + [AIMessage(content=result)]
        logger.info(f"✅ 智能体执行完成 | 输出预览：{result[:50]}...")

        return {
            "output": result,
            "messages": messages
        }

    except Exception as e:
        # 异常捕获：记录栈信息 + 返回重规划标记
        logger.error(f"❌ 智能体执行失败：{str(e)}", exc_info=True)
        return {
            "messages": [
                AIMessage(
                    content=f"智能体执行失败：{str(e)}",
                    additional_kwargs={"error": traceback.format_exc()}  # 附加完整栈信息
                )
            ],
            "need_replan": True  # 标记需要重规划
        }