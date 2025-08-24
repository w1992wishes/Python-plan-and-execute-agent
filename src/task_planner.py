from state import AgentState, Plan, PlanStep, PlanType
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import json
import re
import time
from logger_config import logger
from prompt_setting import get_planning_system_prompt, create_planning_prompt
from settings import Settings
from agent_tools import get_all_tools, get_tools_map
from langchain_openai import ChatOpenAI
from message_parser import parse_messages
from langchain.tools.render import render_text_description


def get_similar_plans(query: str) -> list:
    """获取与查询相似的历史计划"""
    logger.debug(f"查找与 '{query}' 相似的计划")
    return []  # 实际项目中应连接记忆系统（如向量数据库）


from json_util import extract_json_safely

def _parse_plan(query: str, plan_text: str, available_tools: list) -> Plan:
    """解析LLM返回的计划文本"""
    try:
        # 提取JSON片段（假设LLM按格式返回）
        json_part = re.split(r"</think>", plan_text)[-1].strip()
        plan_data = extract_json_safely(json_part)

        # 验证核心字段
        required_fields = ["id", "query", "goal", "plan_type", "steps"]
        for field in required_fields:
            if field not in plan_data:
                raise ValueError(f"计划数据缺少必要字段: {field}")

        # 构建步骤对象
        steps = []
        for i, step_data in enumerate(plan_data["steps"]):
            step = PlanStep(
                id=step_data.get("id", f"step_{i + 1}"),
                description=step_data.get("description", ""),
                tool=step_data.get("tool", ""),
                tool_args=step_data.get("tool_args", {}),
                input_template=step_data.get("input_template", ""),
                dependencies=step_data.get("dependencies", []),
                expected_output=step_data.get("expected_output", ""),
                confidence=step_data.get("confidence", 0.7)
            )

            # 过滤不可用工具
            if step.tool and step.tool not in available_tools:
                logger.warning(f"计划中存在不可用工具: {step.tool}，已忽略")
                step.tool = ""  # 清空无效工具

            steps.append(step)

        # 构建完整计划对象
        return Plan(
            id=plan_data.get("id", f"plan_{int(time.time())}"),
            query=plan_data.get("query", query),
            goal=plan_data.get("goal", "任务规划"),
            plan_type=PlanType(plan_data.get("plan_type", "sequential")),
            steps=steps,
            estimated_duration=plan_data.get("estimated_duration", 60.0),
            confidence=plan_data.get("confidence", 0.7),
            metadata=plan_data.get("metadata", {"source": "llm"}),
            created_at=plan_data.get("created_at", time.time())
        )

    except Exception as e:
        logger.error(f"计划解析失败: {e}", exc_info=True)
        raise


def _create_planning_prompt(query: str, tools_str: str, similar_plans: list, context: dict = None) -> str:
    """构建任务规划的Prompt（包含格式约束）"""
    context_str = json.dumps(context, ensure_ascii=False) if context else "无上下文信息"

    # 相似计划格式化
    similar_plans_str = "无相似历史计划"
    if similar_plans:
        similar_plans_str = "\n".join([
            f"- 计划ID: {p.id}, 目标: {p.goal}, 类型: {p.plan_type.value}"
            for p in similar_plans[:3]  # 最多展示3条
        ])

    return f"""### 用户查询
{query}

### 可用工具
{tools_str}

### 上下文信息
{context_str}

### 相似历史计划
{similar_plans_str}

### 任务要求
1. 生成**分步执行计划**，明确每个步骤的依赖关系  
2. 为步骤选择**可用工具**（需在工具列表中），工具入参支持引用前置步骤结果（如 `step_1_data`）  
3. 填写**输入模板**（支持变量插值）、**预期输出**，并评估**步骤置信度**（0.0-1.0）  
4. 整体计划需标注**类型**（sequential/parallel）、**预计耗时**（秒）和**整体置信度**  

### 输出格式（必须严格遵循JSON）
{{
  "id": "唯一计划ID",
  "query": "用户原始查询",
  "goal": "计划目标",
  "plan_type": "sequential/parallel",
  "steps": [
    {{
      "id": "step_1",
      "description": "步骤描述",
      "tool": "工具名称",
      "tool_args": "工具入参（支持引用前置步骤）",
      "input_template": "输入模板（如 {{city}} 的天气）",
      "dependencies": ["step_1"],  # 依赖的步骤ID列表
      "expected_output": "预期输出描述",
      "confidence": 0.8
    }}
  ],
  "estimated_duration": 60,
  "confidence": 0.8,
  "created_at": 1690000000  # 时间戳
}}"""


class PlanGenerator:
    """计划生成器（封装LLM调用和计划解析）"""

    def __init__(self):
        self.tools = get_all_tools()  # 获取所有可用工具
        self.tools_str = render_text_description(self.tools)  # 渲染工具描述
        self.tools_map = get_tools_map()  # 工具映射表

        # 初始化LLM
        self.model = ChatOpenAI(
            model=Settings.LLM_MODEL,
            temperature=Settings.TEMPERATURE,
            api_key=Settings.OPENAI_API_KEY,
            base_url=Settings.OPENAI_BASE_URL,
        )

    def generate_plan(self, query: str, intent_type: str, context: dict) -> Plan:
        """生成完整执行计划"""
        # 获取相似历史计划（模拟，实际需对接记忆系统）
        similar_plans = get_similar_plans(query)

        # 构建Prompt
        prompt = create_planning_prompt(
            query=query,
            tools_str=self.tools_str,
            similar_plans=similar_plans,
            context=context
        )

        # 调用LLM
        messages = [
            SystemMessage(content=get_planning_system_prompt(intent_type=intent_type)),
            HumanMessage(content=prompt)
        ]
        response = self.model.invoke(messages)
        parse_messages([response])  # 解析并打印消息

        # 解析计划
        return _parse_plan(
            query=query,
            plan_text=response.content,
            available_tools=[t.name for t in self.tools]
        )


def task_planner_node(state: AgentState) -> dict:
    """LangGraph节点：生成任务执行计划"""
    logger.info("🚦✨ [任务规划节点启动，开始生成执行计划...]")

    try:
        # 初始化生成器并创建计划
        generator = PlanGenerator()
        plan = generator.generate_plan(
            query=state["input"],
            intent_type=state.get("intent_type", "unknown"),
            context=state.get("context", {})
        )

        # 记录成功日志
        logger.info(f"✅ 计划生成成功 | ID: {plan.id} | 步骤数: {len(plan.steps)} | 置信度: {plan.confidence}")
        print(f"Generated Plan:\n{plan}")

        # 更新状态
        return {
            "current_plan": plan,
            "plan_history": state.get("plan_history", []) + [plan],
            "need_replan": False,
            "messages": state.get("messages", []) + [
                AIMessage(content=f"计划生成成功！具体信息：\n{plan}")
            ]
        }

    except Exception as e:
        # 异常处理：生成应急计划
        logger.exception(f"❌ 计划生成失败: {str(e)}")

        # 构造应急计划（默认走意图分类）
        default_plan = Plan(
            id=f"emergency_plan_{int(time.time())}",
            query=state["input"],
            goal="应急处理（ fallback ）",
            plan_type=PlanType.SEQUENTIAL,
            steps=[
                PlanStep(
                    id="step_1",
                    description="紧急意图分类",
                    tool="intent_classifier",
                    tool_args={},
                    input_template="{query}",
                    dependencies=[],
                    expected_output="意图分类结果",
                    confidence=0.5
                )
            ],
            estimated_duration=30.0,
            confidence=0.4,
            metadata={"error": str(e), "fallback": True},
            created_at=time.time()
        )

        # 更新失败状态
        return {
            "current_plan": default_plan,
            "plan_history": state.get("plan_history", []) + [default_plan],
            "need_replan": True,
            "messages": [
                AIMessage(content=f"计划生成异常（原因：{str(e)}），已启用应急计划")
            ]
        }