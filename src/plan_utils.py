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
from typing import Dict, Any, List


def extract_json_safely(json_str: str) -> Dict[str, Any]:
    """安全解析JSON字符串（处理常见格式错误）"""
    try:
        # 移除可能的代码块包裹（如 ```json ... ``` 或 </think>...</think>）
        json_str = re.sub(r"^```(?:json)?\s*|\s*```$", "", json_str.strip())
        json_str = re.sub(r"^</think>\s*|\s*</think>$", "", json_str.strip())
        # 修复尾逗号问题（JSON不允许尾逗号）
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
        # 修复单引号问题（JSON要求双引号）
        json_str = re.sub(r"(?<!\\)'", '"', json_str)
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        error_msg = f"JSON解析失败：{str(e)}，原始内容（前100字符）：{json_str[:100]}..."
        logger.warning(error_msg)
        return {"error": error_msg, "raw_content": json_str[:200]}


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
        # 模拟逻辑：实际场景需替换为向量检索（如Pinecone/Chroma）
        # 此处返回空列表，可根据需求扩展历史计划复用逻辑
        return []

    def _build_llm_messages(self, system_prompt: str, user_prompt: str) -> List[Any]:
        """构建LLM输入消息（固定System + Human结构，确保格式统一）"""
        return [
            SystemMessage(content=system_prompt.strip()),
            HumanMessage(content=user_prompt.strip())
        ]

    def _parse_llm_response(self, query: str, response_content: str) -> Plan:
        """解析LLM返回的计划内容，生成Plan对象（含错误降级）"""
        try:
            # 1. 提取并安全解析JSON
            plan_data = extract_json_safely(response_content)
            if "error" in plan_data:
                raise ValueError(f"JSON解析异常：{plan_data['error']}")

            # 2. 构建步骤列表（处理LLM返回的步骤数据）
            steps = []
            for idx, step_data in enumerate(plan_data.get("steps", [])):
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

            # 3. 构建完整计划对象
            plan = Plan(
                id=plan_data.get("id", f"plan_{int(time.time())}"),
                query=plan_data.get("query", query),
                goal=plan_data.get("goal", f"处理用户查询：{query[:30]}..."),
                plan_type=PlanType(plan_data.get("plan_type", "sequential").lower()),
                steps=steps,
                estimated_duration=max(plan_data.get("estimated_duration", 60.0), 10.0),  # 最小10秒（避免不合理值）
                confidence=min(max(plan_data.get("confidence", 0.7), 0.1), 1.0),
                metadata={
                    "generated_by": Settings.LLM_MODEL,
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "tool_count": len(self.tools),
                    **plan_data.get("metadata", {})  # 合并LLM返回的元数据
                },
                created_at=plan_data.get("created_at", time.time())
            )

            # 4. 校验计划兼容性（警告但不阻断，避免过度严格导致流程中断）
            valid, msg = validate_plan_for_react(plan)
            if not valid:
                logger.warning(f"计划兼容性校验未通过：{msg}，将尝试执行（建议优化）")

            return plan

        except Exception as e:
            # 降级逻辑：生成应急计划（确保流程不中断）
            error_msg = f"LLM计划解析失败：{str(e)}"
            logger.error(error_msg, exc_info=True)

            emergency_plan = Plan(
                id=f"emergency_plan_{int(time.time())}",
                query=query,
                goal="应急处理：计划解析失败后的降级流程",
                plan_type=PlanType.SEQUENTIAL,
                steps=[
                    PlanStep(
                        id=f"emergency_step_1_{int(time.time() % 1000)}",
                        description="重新执行意图分类（计划解析失败降级）",
                        tool="intent_classifier",
                        tool_args={"query": query},
                        input_template="基于用户查询'{query}'重新分类意图",
                        dependencies=[],
                        expected_output="获取准确的意图类型（SIMPLE_QUERY/COMPARISON/ROOT_CAUSE_ANALYSIS）",
                        confidence=0.5
                    ),
                    PlanStep(
                        id=f"emergency_step_2_{int(time.time() % 1000)}",
                        description="使用搜索工具获取基础信息（应急方案）",
                        tool="tavily_search" if "tavily_search" in self.tool_names else "",
                        tool_args={"query": query},
                        input_template="搜索关键词：{query}",
                        dependencies=["emergency_step_1"],
                        expected_output="获取与查询相关的基础信息（用于后续回答）",
                        confidence=0.4
                    )
                ],
                estimated_duration=120.0,  # 应急计划预留更长时间
                confidence=0.4,
                metadata={
                    "error": error_msg,
                    "fallback": True,
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                created_at=time.time()
            )

            return emergency_plan

    def generate(self, system_prompt: str, user_prompt: str, query: str) -> Plan:
        """通用计划生成入口（调用LLM + 解析结果）"""
        try:
            # 1. 构建LLM输入消息
            messages = self._build_llm_messages(system_prompt, user_prompt)
            logger.debug(f"[计划生成器] 调用LLM，消息数：{len(messages)}，查询预览：{query[:30]}...")

            # 2. 调用LLM获取计划文本
            response = self.llm.invoke(messages)
            if not hasattr(response, "content") or not response.content:
                raise ValueError("LLM返回空内容，无法生成计划")

            # 3. 解析LLM响应并生成Plan对象
            return self._parse_llm_response(query, response.content)

        except Exception as e:
            error_msg = f"计划生成失败：{str(e)}"
            logger.error(error_msg, exc_info=True)
            # 降级：返回应急计划
            return self._parse_llm_response(query, f'{{"error": "{error_msg}", "steps": []}}')