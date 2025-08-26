from state import Plan, PlanStep, PlanType, AgentState
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools.render import render_text_description
from logger_config import logger
from settings import Settings
from agent_tools import get_all_tools
import re
import time
import json
from typing import Dict, Any, List, Optional


def extract_json_safely(input_str: str) -> Optional[Dict[str, Any]]:
    """
    安全提取JSON：兼容纯JSON字符串和Markdown JSON代码块，返回None时明确日志
    :param input_str: 输入字符串（可能是纯JSON，也可能是带```json标记的代码块）
    :return: 解析后的JSON对象（dict），失败则返回None并记录日志
    """
    # 第一步：先尝试直接解析（处理纯JSON字符串场景）
    try:
        clean_str = input_str.strip()
        if not clean_str:
            logger.warning("[JSON提取] 输入字符串为空，无法解析")
            return None
        parsed = json.loads(clean_str)
        # 确保解析结果是字典（计划格式要求）
        if not isinstance(parsed, dict):
            logger.warning(f"[JSON提取] 解析结果非字典类型（{type(parsed).__name__}），不符合计划格式")
            return None
        return parsed
    except json.JSONDecodeError as e:
        logger.debug(f"[JSON提取] 直接解析失败（{str(e)[:50]}），尝试提取Markdown代码块")

    # 第二步：提取Markdown代码块中的JSON内容
    try:
        # 正则匹配：支持```json、```等多种代码块标记（兼容LLM可能的格式）
        pattern = r'```(?:json)?\s*(.*?)\s*```'
        match_result = re.search(pattern, input_str, re.DOTALL)
        if not match_result:
            logger.warning("[JSON提取] 未找到Markdown代码块标记（如```json）")
            return None

        markdown_json = match_result.group(1).strip()
        if not markdown_json:
            logger.warning("[JSON提取] Markdown代码块内内容为空")
            return None

        parsed = json.loads(markdown_json)
        if not isinstance(parsed, dict):
            logger.warning(f"[JSON提取] 代码块内解析结果非字典类型（{type(parsed).__name__}）")
            return None
        return parsed
    except json.JSONDecodeError as e:
        logger.error(f"[JSON提取] Markdown代码块解析失败：{str(e)}，原始内容：{markdown_json[:100]}...")
    except Exception as e:
        logger.error(f"[JSON提取] 未知异常：{str(e)}", exc_info=True)

    return None


def validate_plan_for_react(plan: Plan) -> tuple[bool, str]:
    """校验计划是否符合ReAct执行器要求"""
    # 1. 校验计划核心字段
    if not plan.id.startswith("plan_"):
        return False, f"计划ID格式错误：{plan.id}（需以'plan_'开头，如 plan_1712345678）"
    if plan.plan_type != PlanType.SEQUENTIAL:
        return False, f"当前仅支持顺序执行计划（sequential），当前类型：{plan.plan_type.value}"
    if not plan.steps:
        return False, "计划不能为空（需至少包含1个步骤）"

    # 2. 校验每个步骤的格式与兼容性
    step_ids = [step.id for step in plan.steps]
    for step in plan.steps:
        # 步骤ID格式校验
        if not step.id.startswith("step_"):
            return False, f"步骤ID格式错误：{step.id}（需以'step_'开头，如 step_1）"
        # 步骤ID唯一性校验
        if step_ids.count(step.id) > 1:
            return False, f"步骤ID重复：{step.id}（所有步骤ID必须唯一）"
        # 工具参数格式校验（ReAct执行器要求JSON对象）
        if not isinstance(step.tool_args, dict):
            return False, f"步骤{step.id}的tool_args格式错误（需为JSON对象，当前：{type(step.tool_args).__name__}）"
        # 依赖步骤有效性校验
        for dep in step.dependencies:
            if dep not in step_ids:
                return False, f"步骤{step.id}依赖无效步骤：{dep}（未在计划中找到该步骤ID）"
        # 置信度范围校验（0.0-1.0）
        if not (0.0 <= step.confidence <= 1.0):
            return False, f"步骤{step.id}的置信度无效：{step.confidence}（需在0.0-1.0范围内）"

    # 3. 校验整体计划置信度
    if not (0.0 <= plan.confidence <= 1.0):
        return False, f"计划整体置信度无效：{plan.confidence}（需在0.0-1.0范围内）"

    return True, "计划符合ReAct执行器要求"


class BasePlanGenerator:
    """基础计划生成器（规划/重规划共用父类）"""

    def __init__(self):
        # 初始化工具列表与元数据
        self.tools = get_all_tools()
        self.tool_names = [tool.name for tool in self.tools]  # 工具名称列表（用于校验）
        self.tools_str = render_text_description(self.tools)  # 工具描述文本（供LLM参考）

        # 初始化LLM客户端（与全局配置对齐）
        self.llm = ChatOpenAI(
            model=Settings.LLM_MODEL,
            temperature=Settings.TEMPERATURE,
            api_key=Settings.OPENAI_API_KEY,
            base_url=Settings.OPENAI_BASE_URL,
            timeout=30,  # 超时保护（避免长期阻塞）
            max_retries=2  # 重试机制（增强稳定性）
        )

    def get_similar_plans(self, query: str) -> List[Plan]:
        """获取相似历史计划（模拟实现，实际需对接向量数据库）"""
        logger.debug(f"[计划工具] 查找与 '{query[:30]}...' 相似的历史计划")
        return []  # 模拟返回空列表

    def _build_llm_messages(self, system_prompt: str, user_prompt: str) -> List[Any]:
        """构建LLM输入消息（固定System + Human结构，确保格式统一）"""
        return [
            SystemMessage(content=system_prompt.strip()),
            HumanMessage(content=user_prompt.strip())
        ]

    def _parse_llm_response(self, query: str, response_content: str) -> Plan:
        """解析LLM返回的计划内容，生成Plan对象（含错误降级）"""
        try:
            # 1. 提取并安全解析JSON（关键修复：增加plan_data空值判断）
            plan_data = extract_json_safely(response_content)
            if plan_data is None:
                raise ValueError("LLM返回内容无法解析为JSON字典（可能格式错误或空内容）")

            # 2. 构建步骤列表（处理LLM返回的步骤数据）
            steps = []
            # 即使plan_data有值，也需判断"steps"是否存在（避免KeyError）
            llm_steps = plan_data.get("steps", [])
            for idx, step_data in enumerate(llm_steps):
                # 补全步骤默认值（避免字段缺失）
                step_id = step_data.get("id", f"step_{idx + 1}_{int(time.time() % 1000)}")
                tool_name = step_data.get("tool", "")

                # 过滤未启用的工具（避免执行时工具不存在）
                if tool_name and tool_name not in self.tool_names:
                    logger.warning(f"步骤{step_id}引用未启用工具：{tool_name}，已清空工具配置")
                    tool_name = ""
                    tool_args = {}
                else:
                    tool_args = step_data.get("tool_args", {})

                # 构建单个步骤对象
                steps.append(PlanStep(
                    id=step_id,
                    description=step_data.get("description", f"未命名步骤（{idx + 1}）"),
                    tool=tool_name,
                    tool_args=tool_args,
                    input_template=step_data.get("input_template", f"基于查询'{query[:20]}...'执行步骤"),
                    dependencies=step_data.get("dependencies", []),
                    expected_output=step_data.get("expected_output", "未定义预期输出"),
                    confidence=min(max(step_data.get("confidence", 0.7), 0.1), 1.0)  # 置信度范围限制
                ))

            # 3. 构建完整计划对象（补全计划默认值，避免字段缺失）
            plan_id = plan_data.get("id", f"plan_{int(time.time())}")
            plan_goal = plan_data.get("goal", f"处理用户查询：{query[:30]}...")
            plan_type = PlanType(plan_data.get("plan_type", "sequential").lower())
            estimated_duration = max(plan_data.get("estimated_duration", 60.0), 10.0)
            plan_confidence = min(max(plan_data.get("confidence", 0.7), 0.1), 1.0)

            plan = Plan(
                id=plan_id,
                query=plan_data.get("query", query),
                goal=plan_goal,
                plan_type=plan_type,
                steps=steps,
                estimated_duration=estimated_duration,
                confidence=plan_confidence,
                metadata={
                    "generated_by": Settings.LLM_MODEL,
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "tool_count": len(self.tools),
                    **plan_data.get("metadata", {})  # 合并LLM返回的元数据
                },
                created_at=plan_data.get("created_at", time.time())
            )

            # 4. 校验计划兼容性（警告但不阻断）
            valid, msg = validate_plan_for_react(plan)
            if not valid:
                logger.warning(f"计划兼容性校验未通过：{msg}，将尝试执行（建议优化）")

            return plan

        except Exception as e:
            # 降级逻辑：生成应急计划（确保流程不中断）
            error_msg = f"LLM计划解析失败：{str(e)}"
            logger.error(error_msg, exc_info=True)

            # 应急计划：优先使用已启用的工具（避免引用不存在的工具）
            emergency_tool = "metric_query" if "metric_query" in self.tool_names else "calculate"
            emergency_steps = [
                PlanStep(
                    id=f"emergency_step_1_{int(time.time() % 1000)}",
                    description="应急：直接调用核心工具获取数据",
                    tool=emergency_tool,
                    tool_args={"query": query},
                    input_template=f"使用{emergency_tool}工具处理查询：{query}",
                    dependencies=[],
                    expected_output=f"通过{emergency_tool}工具获取的基础数据",
                    confidence=0.5
                )
            ]

            # 若没有任何启用工具，生成无工具的应急步骤
            if not self.tool_names:
                emergency_steps = [
                    PlanStep(
                        id=f"emergency_step_1_{int(time.time() % 1000)}",
                        description="应急：无可用工具，直接返回查询建议",
                        tool="",
                        tool_args={},
                        input_template=f"分析用户查询：{query}",
                        dependencies=[],
                        expected_output="基于查询的自然语言建议",
                        confidence=0.3
                    )
                ]

            emergency_plan = Plan(
                id=f"emergency_plan_{int(time.time())}",
                query=query,
                goal="应急处理：计划解析失败后的降级流程",
                plan_type=PlanType.SEQUENTIAL,
                steps=emergency_steps,
                estimated_duration=120.0,
                confidence=0.4,
                metadata={
                    "error": error_msg,
                    "fallback": True,
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                created_at=time.time()
            )

            return emergency_plan