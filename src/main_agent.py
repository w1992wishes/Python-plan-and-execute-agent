import asyncio
from src.workflow.graph_builder import create_async_agent_workflow
from src.workflow.agent_tools import get_tools_map
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from setting.settings import Settings
from log.logger import logger
from src.workflow.state import AgentState


class MultiStepAgent:
    def __init__(self, temperature: float = None):
        """初始化异步多步任务Agent"""
        # 加载LLM（优先使用传入温度，无则用配置）
        self.model = ChatOpenAI(
            model=Settings.LLM_MODEL,
            temperature=temperature or Settings.TEMPERATURE,
            api_key=Settings.OPENAI_API_KEY,
            base_url=Settings.OPENAI_BASE_URL,
            timeout=30,
            max_retries=2
        )
        self.tools_map = get_tools_map()  # 工具映射表（供日志参考）
        # 加载异步工作流（需确保create_agent_workflow返回异步CompiledStateGraph）
        self.graph = create_async_agent_workflow()
        logger.info("✅ 异步MultiStepAgent初始化完成（LLM：%s，工具数：%d）",
                    Settings.LLM_MODEL, len(self.tools_map))

    async def run_async(self, query: str) -> str:
        """
        异步执行查询：迭代工作流事件流，实时打印节点结果，返回最终回复
        核心：通过graph.astream()获取异步事件，按指定格式输出
        """
        # 1. 初始化AgentState（纯类对象，与异步工作流兼容）
        initial_state = AgentState(
            input=query,
            messages=[HumanMessage(content=query)],  # 初始化用户消息
            intent_type="SIMPLE_QUERY"
        )
        logger.info(f"📥 接收查询：{query[:50]}...")

        final_response = "无回复"  # 默认最终回复

        # 2. 异步迭代工作流事件（核心逻辑：实时捕获节点执行结果）
        async for event in self.graph.astream(initial_state):
            # event结构：{节点名称: 节点执行后的AgentState}
            for node_name, node_output in event.items():
                # 🔴 处理工作流结束节点
                if node_name == "__end__":
                    print("\n" + "="*50)
                    print("=== 工作流结束 ===")
                    # 提取最终回复（取最后一条AIMessage）
                    aimessages = [msg for msg in node_output.messages if isinstance(msg, AIMessage)]
                    if aimessages:
                        final_response = aimessages[-1].content[:200] + "..." if len(aimessages[-1].content) > 200 else aimessages[-1].content
                    print(f"最终回复：{final_response}")
                    print("="*50)
                    continue

                # 🟢 处理普通节点（意图分类/规划/执行/重规划）
                print("\n" + "="*50)
                print(f"=== 节点 {node_name} 执行结果 ===")

                # 👇 按用户要求格式打印关键信息（字段映射：AgentState → 输出格式）
                # 1. 打印更新后的计划（对应AgentState.current_plan）
                if "current_plan" in node_output and node_output.get("current_plan"):
                    plan = node_output.get("current_plan")
                    print("更新后的计划：")
                    print(f"  计划ID：{plan.id[:12]}... | 目标：{plan.goal[:30]}...")
                    print("  步骤列表：")
                    for idx, step in enumerate(plan.steps, 1):
                        print(f"    {idx}. 工具：{step.tool or '无'} | 描述：{step.description[:40]}...")

                # 2. 打印已完成步骤（对应AgentState.executed_steps）
                if "executed_steps" in node_output and node_output.get("executed_steps"):
                    # 取最后一个已完成步骤
                    last_step = node_output.get("executed_steps")[-1]
                    task = f"步骤{last_step['step_id']}-{last_step['description']}（工具：{last_step['tool_used'] or '无'}）"
                    result = last_step['result']
                    print(f"已完成步骤：{task}")
                    print(f"步骤结果：{result[:100]}..." if len(str(result)) > 100 else f"步骤结果：{result}")

                # 3. 打印节点直接回复（对应AgentState.messages的最后一条AIMessage）
                if "messages" in node_output and node_output.get("messages"):
                    last_msg = node_output.get("messages")[-1]
                    if isinstance(last_msg, AIMessage) and "回复" in last_msg.content[:20]:
                        print(f"直接回复：{last_msg.content[:80]}...")

                print("="*50)

        return final_response

    async def chat_async(self):
        """启动异步交互式聊天会话"""
        print("\n" + "="*50)
        print("🎯 ReAct Agent 异步聊天启动")
        print("📌 输入 'exit'/'quit' 退出对话 | 输入 'clear' 清空屏幕")
        print("="*50)

        while True:
            try:
                # 接收用户输入（异步环境中保持同步输入，避免复杂IO）
                query = input("\n你：").strip()

                # 处理命令：退出/清空
                if query.lower() in ["exit", "quit"]:
                    print("\nAgent：感谢使用，再见！")
                    logger.info("👋 聊天会话结束")
                    break
                if query.lower() == "clear":
                    print("\033c", end="")  # 清空屏幕（兼容大多数终端）
                    continue
                if not query:
                    print("Agent：请输入具体问题，我会帮你处理～")
                    continue

                # 异步执行查询并获取结果
                await self.run_async(query)

            except KeyboardInterrupt:
                print("\n\nAgent：检测到中断信号，正在安全退出...")
                logger.warning("⚠️  聊天被键盘中断")
                break
            except Exception as e:
                error_msg = f"处理请求失败：{str(e)[:50]}..."
                logger.error(f"❌ 聊天会话异常：{str(e)}", exc_info=True)
                print(f"Agent：{error_msg} 请重新输入问题～")


async def main():
    """异步程序入口：初始化Agent并启动聊天"""
    try:
        # 初始化Agent（温度设为0.1，平衡准确性与灵活性）
        agent = MultiStepAgent(temperature=0.1)
        # 启动异步聊天
        await agent.chat_async()
    except Exception as e:
        logger.critical(f"❌ Agent启动失败：{str(e)}", exc_info=True)
        print(f"\nAgent启动失败：{str(e)}，请检查配置后重试～")


if __name__ == "__main__":
    # 启动异步主函数（Python异步入口标准写法）
    asyncio.run(main())