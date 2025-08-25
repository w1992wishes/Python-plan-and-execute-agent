from graph_builder import create_agent_workflow
from agent_tools import get_tools_map
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from settings import Settings
from logger_config import logger
from state import AgentState


class MultiStepAgent:
    def __init__(self, temperature: float = 0):
        """初始化多步任务 Agent"""
        # 加载模型（优先使用传入的 temperature，否则读取配置）
        self.model = ChatOpenAI(
            model=Settings.LLM_MODEL,
            temperature=temperature or Settings.TEMPERATURE,
            api_key=Settings.OPENAI_API_KEY,
            base_url=Settings.OPENAI_BASE_URL,
        )
        self.tools = get_tools_map()  # 工具映射表
        self.graph = create_agent_workflow()  # 加载状态机图

    def run(self, query: str) -> str:
        """执行单次查询，返回最终响应"""
        initial_state = AgentState(
            input=query,
            messages=[],
            intent_type="SIMPLE_QUERY"
        )
        result = self.graph.invoke(initial_state)  # 执行状态机
        return result["messages"][-1].content  # 提取最终回复

    def chat(self):
        """启动交互式聊天会话"""
        print("=" * 50)
        print("Agent: 你好！我是多步任务处理助手")
        print("输入 'exit' 或 'quit' 退出对话")
        print("=" * 50)

        conversation_history = []  # 对话历史存储

        while True:
            try:
                query = input("\n你：").strip()

                # 处理退出命令
                if query.lower() in ["exit", "quit"]:
                    print("\nAgent: 感谢使用，再见！")
                    break

                # 处理空输入
                if not query:
                    print("Agent: 请告诉我您需要什么帮助？")
                    continue

                # 执行查询并获取响应
                response = self.run(query)

                # 记录对话历史
                conversation_history.append(f"你：{query}")
                conversation_history.append(f"Agent：{response}")

                # 输出响应
                print(f"\nAgent：{response}")
                print("-" * 50)

            except KeyboardInterrupt:
                print("\n\nAgent：检测到中断信号，正在退出...")
                break
            except Exception as e:
                logger.error(f"发生异常：{e}", exc_info=True)
                print("请重新输入您的问题")


def main():
    """程序入口：初始化 Agent 并启动聊天"""
    agent = MultiStepAgent(temperature=0)  # 初始化 Agent（可调整 temperature）

    print("=" * 50)
    print("ReAct Agent 系统已启动（输入 'exit' 退出）")
    print("=" * 50)
    agent.chat()  # 启动交互式聊天


if __name__ == "__main__":
    main()