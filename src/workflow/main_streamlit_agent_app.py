import asyncio
import streamlit as st
from src.workflow.graph_builder import create_async_agent_workflow
from src.workflow.agent_tools import get_tools_map
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from src.setting.settings import Settings
from src.log.logger import logger
from src.workflow.state import AgentState

st.set_page_config(
    page_title="ReAct Agent 多步任务执行平台",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource(show_spinner="正在初始化 Agent 组件...")
def init_agent():
    try:
        llm = ChatOpenAI(
            model=Settings.LLM_MODEL,
            temperature=0.1,
            api_key=Settings.OPENAI_API_KEY,
            base_url=Settings.OPENAI_BASE_URL,
            timeout=30,
            max_retries=2
        )
        tools_map = get_tools_map()
        workflow = create_async_agent_workflow()
        logger.info(f"✅ Agent 初始化完成（LLM：{Settings.LLM_MODEL}，工具数：{len(tools_map)}）")
        return {
            "llm": llm,
            "tools_map": tools_map,
            "workflow": workflow
        }
    except Exception as e:
        st.error(f"❌ Agent 初始化失败：{str(e)}")
        logger.critical(f"Agent 初始化崩溃：{str(e)}", exc_info=True)
        raise


async def run_agent_async(query: str, status_placeholder, log_placeholder):
    # 初始化日志列表 - 使用会话状态保存日志，避免丢失
    st.session_state.current_logs = []
    logs = st.session_state.current_logs

    # 初始化状态
    initial_state = AgentState(
        input=query,
        messages=[HumanMessage(content=query)],
        intent_type="SIMPLE_QUERY"
    )
    final_response = "未获取到有效回复"
    agent_resources = init_agent()
    workflow = agent_resources["workflow"]

    # 初始日志
    logs.append("📝 执行日志（实时更新）")
    logs.append(f"> **查询内容**：{query[:100]}{'...' if len(query) > 100 else ''}")
    log_placeholder.markdown("\n\n".join(logs))

    # 遍历工作流事件
    async for event in workflow.astream(initial_state):
        for node_name, node_output in event.items():
            # 统一 node_output 为字典格式
            if not isinstance(node_output, dict):
                try:
                    node_output = node_output.dict()
                except:
                    node_output = vars(node_output)

            # 处理结束节点
            if node_name == "__end__":
                # 更新状态为完成
                status_placeholder.success("🎉 工作流执行完成！")

                # 添加总结日志
                logs.append("\n---")
                logs.append("### 📊 执行总结")

                # 提取最终回复
                messages = node_output.get("messages", [])
                aimessages = [msg for msg in messages if isinstance(msg, AIMessage)]
                if aimessages:
                    final_response = aimessages[-1].content
                    logs.append(f"> **最终回复**：{final_response[:200]}{'...' if len(final_response) > 200 else ''}")
                else:
                    executed_steps = node_output.get("executed_steps", [])
                    if executed_steps:
                        final_response = f"任务已完成，共执行 {len(executed_steps)} 步。最后一步结果：{str(executed_steps[-1].get('result', ''))[:100]}"
                        logs.append(f"> **最终回复**：{final_response}")

                # 统计步骤
                executed_steps = node_output.get("executed_steps", [])
                logs.append(f"> **总执行步骤**：{len(executed_steps)} 个")
                logs.append(f"> **剩余步骤**：{len(node_output.get('current_plan', {}).get('steps', []))} 个")

                # 更新最终日志
                log_placeholder.markdown("\n\n".join(logs))
                return final_response  # 直接返回，避免重复处理

            # 处理普通节点 - 先更新状态
            status_placeholder.info(f"🔄 当前节点：{node_name}（正在执行...）")

            # 添加节点日志
            logs.append("\n---")
            logs.append(f"### 🔧 节点：{node_name}")

            # 打印计划信息
            current_plan = node_output.get("current_plan", {})
            if current_plan:
                try:
                    plan_id = current_plan.id if hasattr(current_plan, 'id') else current_plan.get('id', '未知ID')
                    plan_goal = current_plan.goal if hasattr(current_plan, 'goal') else current_plan.get('goal',
                                                                                                         '无目标')
                    logs.append(
                        f"> **计划信息**：ID={plan_id[:12]}... | 目标={plan_goal[:60]}{'...' if len(plan_goal) > 60 else ''}")

                    steps = current_plan.steps if hasattr(current_plan, 'steps') else current_plan.get('steps', [])
                    if steps:
                        logs.append("> **当前步骤列表**：")
                        for idx, step in enumerate(steps, 1):
                            step_tool = step.tool if hasattr(step, 'tool') else step.get('tool', '无工具')
                            step_desc = step.description if hasattr(step, 'description') else step.get('description',
                                                                                                       '无描述')
                            logs.append(
                                f"  {idx}. 工具：{step_tool} | 描述：{step_desc[:50]}{'...' if len(step_desc) > 50 else ''}")
                    else:
                        logs.append("> **当前步骤列表**：无待执行步骤")
                except Exception as e:
                    logs.append(f"> **计划信息解析错误**：{str(e)}")
            else:
                logs.append("> **计划信息**：暂无计划数据")

            # 打印已完成步骤
            executed_steps = node_output.get("executed_steps", [])
            if executed_steps:
                last_step = executed_steps[-1]
                step_id = last_step.get("step_id", "未知ID")
                step_tool = last_step.get("tool_used", "无工具")
                step_desc = last_step.get("description", "无描述")
                step_result = str(last_step.get("result", "无结果"))
                logs.append(f"> **最新完成步骤**：{step_id}（工具：{step_tool}）")
                logs.append(f"> **步骤描述**：{step_desc[:80]}{'...' if len(step_desc) > 80 else ''}")
                logs.append(f"> **步骤结果**：{step_result[:150]}{'...' if len(step_result) > 150 else ''}")
            else:
                logs.append("> **已完成步骤**：暂无执行记录")

            # 打印节点消息
            messages = node_output.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, AIMessage):
                    logs.append(
                        f"> **节点消息**：{last_msg.content[:100]}{'...' if len(last_msg.content) > 100 else ''}")
                elif isinstance(last_msg, HumanMessage):
                    logs.append(
                        f"> **用户消息**：{last_msg.content[:100]}{'...' if len(last_msg.content) > 100 else ''}")
            else:
                logs.append("> **节点消息**：无消息")

            # 更新日志显示
            log_placeholder.markdown("\n\n".join(logs))

    return final_response


def main():
    # 初始化会话状态 - 确保所有状态变量都被正确初始化
    if "history" not in st.session_state:
        st.session_state.history = []
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "current_logs" not in st.session_state:
        st.session_state.current_logs = []
    if "last_executed_query" not in st.session_state:
        st.session_state.last_executed_query = ""

    st.title("🤖 ReAct Agent 多步任务执行平台")
    st.caption("实时展示每一步执行结果（包含节点详情、步骤数据）")
    st.divider()

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("📥 输入查询")
        user_query = st.text_area(
            label="任务需求示例：查询河南上月客户量并与深圳对比",
            height=150,
            disabled=st.session_state.is_running,
            placeholder="支持多步任务、指标查询、数学计算等..."
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            submit_btn = st.button(
                "🚀 执行任务",
                type="primary",
                disabled=not user_query or st.session_state.is_running
            )
        with col_btn2:
            clear_btn = st.button(
                "🗑️ 清空记录",
                disabled=st.session_state.is_running
            )

        # 清空历史记录和当前日志
        if clear_btn:
            st.session_state.history = []
            st.session_state.current_logs = []
            st.session_state.last_executed_query = ""
            # 使用rerun而不是直接清空，确保UI正确更新
            st.experimental_rerun()

        # 显示历史记录
        if st.session_state.history:
            st.subheader("📜 历史记录")
            for idx, (q, a) in enumerate(reversed(st.session_state.history), 1):
                with st.expander(f"历史 {idx}：{q[:30]}{'...' if len(q) > 30 else ''}"):
                    st.markdown(f"**回复**：{a}")

    with col2:
        # 状态容器 - 使用占位符确保状态能被更新
        st.subheader("🔍 执行状态")
        status_placeholder = st.empty()

        # 日志容器
        st.subheader("📝 执行日志")
        log_placeholder = st.empty()

        # 显示状态和日志 - 根据不同情况显示不同内容
        if st.session_state.is_running:
            # 执行中，状态已经由run_agent_async管理
            pass
        elif st.session_state.current_logs:
            # 执行完成且有日志，显示最后状态和日志
            status_placeholder.success("🎉 工作流执行完成！")
            log_placeholder.markdown("\n\n".join(st.session_state.current_logs))
        else:
            # 未执行过任务，显示初始状态
            status_placeholder.info("等待输入任务并点击「执行任务」")
            log_placeholder.markdown("""
            > 执行时将显示：
            > 1. 每个节点的详细执行数据
            > 2. 已完成步骤的结果
            > 3. 实时更新的计划信息
            """)

    # 执行任务逻辑
    if submit_btn and not st.session_state.is_running:
        st.session_state.is_running = True
        st.session_state.last_executed_query = user_query  # 保存当前执行的查询

        # 重置状态显示
        status_placeholder.info("🔄 准备执行任务...")
        log_placeholder.text("初始化执行环境...")

        try:
            # 执行任务并获取最终回复
            final_response = asyncio.run(run_agent_async(
                query=user_query,
                status_placeholder=status_placeholder,
                log_placeholder=log_placeholder
            ))

            # 保存到历史记录
            st.session_state.history.append((user_query, final_response))

        except Exception as e:
            error_msg = f"❌ 执行失败：{str(e)[:100]}"
            final_response = error_msg
            status_placeholder.error(error_msg)
            st.session_state.current_logs.append(f"> **错误详情**：{str(e)}")
            log_placeholder.markdown("\n\n".join(st.session_state.current_logs))
            st.session_state.history.append((user_query, final_response))
            logger.error(f"执行异常：{str(e)}", exc_info=True)
        finally:
            st.session_state.is_running = False
            # 不刷新页面，保持内容显示
            # 仅更新状态和日志占位符，不触发页面整体刷新


if __name__ == "__main__":
    main()
