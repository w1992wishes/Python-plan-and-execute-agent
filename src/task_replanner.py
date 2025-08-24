from state import AgentState, Plan, PlanStep, PlanType
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
import json
import re
import time
from logger_config import logger
from settings import Settings
from agent_tools import get_all_tools
from langchain_openai import ChatOpenAI
from typing import Dict, List, Any
from json_util import extract_json_safely


def _parse_replan(plan_text: str, original_plan: Plan, available_tools: list) -> Plan:
    """解析LLM返回的重新规划文本（修复格式错误，优化异常处理）"""
    try:
        plan_data = extract_json_safely(plan_text)

        # 验证核心字段完整性
        required_fields = ["id", "query", "goal", "plan_type", "steps"]
        for field in required_fields:
            if field not in plan_data:
                raise ValueError(f"重新规划数据缺少必要字段: {field}")

        # 构建步骤对象（补全所有必填属性，避免实例化报错）
        steps = []
        for i, step_data in enumerate(plan_data["steps"]):
            step_id = step_data.get("id", f"step_{i + 1}")
            # 确保PlanStep实例化参数完整（适配PlanStep类定义）
            step = PlanStep(
                id=step_id,
                description=step_data.get("description", f"未命名步骤_{step_id}"),
                tool=step_data.get("tool", ""),
                tool_args=step_data.get("tool_args", {}),
                input_template=step_data.get("input_template", ""),
                dependencies=step_data.get("dependencies", []),
                expected_output=step_data.get("expected_output", "未定义预期输出"),
                confidence=step_data.get("confidence", 0.6)
            )

            # 过滤不可用工具
            if step.tool and step.tool not in available_tools:
                logger.warning(f"重新规划中存在不可用工具: {step.tool}，已清空")
                step.tool = ""

            steps.append(step)

        # 构建完整计划（修复PlanType枚举转换，避免类型错误）
        return Plan(
            id=plan_data.get("id", f"replan_{int(time.time())}"),
            query=plan_data.get("query", original_plan.query),
            goal=plan_data.get("goal", original_plan.goal),
            plan_type=PlanType(plan_data.get("plan_type", original_plan.plan_type.value)),
            steps=steps,
            estimated_duration=plan_data.get(
                "estimated_duration",
                int(original_plan.estimated_duration * len(steps) / len(original_plan.steps))
            ),
            confidence=min(
                plan_data.get("confidence", original_plan.confidence * 0.85),
                original_plan.confidence
            ),
            metadata={
                "replan_reason": plan_data.get("replan_reason", "基于执行结果自动调整"),
                "based_on_execution": True,
                "replan_timestamp": int(time.time())
            },
            created_at=time.time()
        )

    except Exception as e:
        logger.error(f"重新规划解析失败: {str(e)}", exc_info=True)
        raise


def _create_replanning_prompt(
        query: str,
        original_plan: Plan,
        final_output: str,
        evaluation: Dict[str, Any],
        tools_str: str
) -> str:
    """构建重新规划的Prompt（修复字符串拼接格式，避免语法错误）"""
    # 1. 解析原始计划步骤信息（修复循环逻辑，确保格式统一）
    original_steps_str = []
    for step in original_plan.steps:
        step_desc = (
            f"步骤 {step.id}: {step.description}\n"
            f" - 工具: {step.tool or '未指定'}\n"
            f" - 依赖: {', '.join(step.dependencies) if step.dependencies else '无'}\n"
            f" - 预期输出: {step.expected_output[:100]}"
        )
        if len(step.expected_output) > 100:
            step_desc += "..."
        original_steps_str.append(step_desc)

    # 2. 解析LLM评估结果（处理空值，避免索引报错）
    need_replan = evaluation.get("need_replan", False)
    reason = evaluation.get("reason", "评估未明确原因")
    issues = evaluation.get("issues", [])
    suggestions = evaluation.get("suggested_adjustments", [])

    # 3. 提取执行结果关键信息（修复截断逻辑）
    truncated_output = final_output[:500]
    if len(final_output) > 500:
        truncated_output += "..."

    # 4. 构建完整Prompt（使用f-string格式化，避免字符串拼接错误）
    return f"""### 任务重规划需求
根据**LLM评估结果**和**执行结果**，生成**调整后的执行计划**，需满足：  
1. 针对性修复评估发现的问题（如结果不相关、分析不完整、存在错误）  
2. 严格遵循原始任务目标，仅调整执行步骤和工具选择  
3. 标注**重规划原因**，新计划需体现对评估建议的响应  
4. 若原始计划有意图类型（如根因分析/对比），需保持该类型的专项要求  


### 原始任务上下文
#### 1. 核心信息  
- 用户原始查询: {query}  
- 原始计划目标: {original_plan.goal}  
- 原始计划类型: {original_plan.plan_type.value}  
- 原始计划步骤数: {len(original_plan.steps)}  
- 原始计划置信度: {original_plan.confidence:.2f}  

#### 2. 原始计划步骤详情  
{"\n\n".join(original_steps_str) if original_steps_str else "无具体步骤"}  


### 执行与评估反馈
#### 1. 执行结果概要  
{truncated_output if truncated_output else "未获取到有效执行结果"}  

#### 2. LLM评估结论  
- 是否需要重规划: {"是" if need_replan else "否"}  
- 重规划核心原因: {reason}  
- 需修复的问题: {'\n  - ' + '\n  - '.join(issues) if issues else "无明确问题"}  
- 评估建议调整方向: {'\n  - ' + '\n  - '.join(suggestions) if suggestions else "无明确建议"}  


### 可用工具约束
以下是仅有的可用工具（必须从列表中选择，不可使用其他工具）：  
{tools_str}  


### 输出格式（必须严格遵循JSON，不可添加额外文本）  
```json  
{{  
  "id": "唯一计划ID（如replan_1712345678）",  
  "query": "用户原始查询（保持不变）",  
  "goal": "调整后的计划目标（与原始目标一致或优化表述）",  
  "plan_type": "sequential（必须为串行，不支持parallel）",  
  "replan_reason": "简要说明重规划原因（关联评估问题）",  
  "steps": [  
    {{  
      "id": "step_1（步骤ID需连续）",  
      "description": "清晰的步骤描述（体现问题修复）",  
      "tool": "工具名称（必须在可用工具列表中）",  
      "tool_args": "工具入参（支持引用前置步骤结果，格式：step_1_result）",  
      "input_template": "具体输入格式（如：查询{{指标名}}在{{时间范围}}的数据）",  
      "dependencies": ["依赖步骤ID列表（如无依赖填[]）"],  
      "expected_output": "明确的预期输出（如：包含{{指标名}}的数值和趋势）",  
      "confidence": 0.8（0-1之间的浮点数，体现步骤可行性）  
    }}  
  ],  
  "estimated_duration": 60（预计总耗时，秒）,  
  "confidence": 0.8（0-1之间的浮点数，新计划整体置信度）,  
  "created_at": {int(time.time())}（当前时间戳）  
}}  
```"""


class ReplanGenerator:
    """重规划生成器（修复工具加载逻辑，确保依赖正确）"""

    def __init__(self):
        # 加载所有可用工具（确保get_all_tools()返回正确的工具列表）
        self.tools = get_all_tools()
        self.available_tools = [tool.name for tool in self.tools]

        # 格式化工具列表（修复工具描述截断逻辑）
        self.tools_str = []
        for tool in self.tools:
            desc = tool.description[:150]
            if len(tool.description) > 150:
                desc += "..."
            self.tools_str.append(f"- {tool.name}: {desc}")
        self.tools_str = "\n".join(self.tools_str)

        # 初始化LLM客户端（修复参数传递，确保与Settings匹配）
        self.model = ChatOpenAI(
            model=Settings.LLM_MODEL,
            temperature=0.4,
            api_key=Settings.OPENAI_API_KEY,
            base_url=Settings.OPENAI_BASE_URL,
            timeout=30  # 新增超时配置，避免无限等待
        )

    def generate_replan(
            self,
            query: str,
            original_plan: Plan,
            final_output: str,
            evaluation: Dict[str, Any]
    ) -> Plan:
        """生成调整后的执行计划（修复参数传递，移除无用依赖）"""
        # 构建重规划Prompt
        prompt = _create_replanning_prompt(
            query=query,
            original_plan=original_plan,
            final_output=final_output,
            evaluation=evaluation,
            tools_str=self.tools_str
        )

        # 构建重规划系统提示（修复字符串格式，避免语法错误）
        system_prompt = """你是专业的任务重规划专家，负责根据执行结果和评估反馈优化任务计划。

重规划核心原则：
1. 问题导向：所有调整必须针对评估发现的问题（如结果不相关→调整查询步骤，分析不完整→补充分析步骤）
2. 工具约束：仅可使用提供的"可用工具"，不可自创工具
3. 逻辑连贯：步骤依赖关系需合理，避免循环依赖
4. 意图适配：
   - SIMPLE_QUERY：确保步骤仅聚焦精准查数，无冗余
   - COMPARISON：保留"查询+对比分析"双步骤结构
   - ROOT_CAUSE_ANALYSIS：必须包含"查数→分析"完整链路，有风险需加下级机构查询步骤

严格按指定JSON格式输出，不添加任何解释性文本。"""

        # 调用LLM生成重规划结果（修复消息结构，确保符合LangChain要求）
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        response = self.model.invoke(messages)
        logger.info(f"重规划LLM响应（前500字符）: \n{response.content[:500]}...")

        # 解析并返回新计划
        return _parse_replan(
            plan_text=response.content,
            original_plan=original_plan,
            available_tools=self.available_tools
        )


def replan_node(state: AgentState) -> dict:
    """LangGraph重规划节点（修复状态提取逻辑，确保类型正确）"""
    logger.info("🔄 重规划节点启动，基于LLM评估结果优化计划...")

    # 1. 从状态中提取核心信息（修复键名匹配，处理空值）
    replan_count = state.get("replan_count", 0)
    current_plan = state.get("current_plan")
    final_output = state.get("output", "")
    evaluation = state.get("evaluation", {})
    existing_messages = state.get("messages", [])
    user_query = state.get("input", "")

    # 2. 基础校验（修复条件判断，避免空对象报错）
    if not current_plan:
        logger.warning("⚠️ 无有效原始计划，无法重规划")
        return {
            "replan_limit": True,
            "replan_count": replan_count,
            "messages": existing_messages + [
                AIMessage(content="重规划失败：未获取到原始执行计划")
            ]
        }
    if not isinstance(evaluation, dict) or not evaluation:
        logger.warning("⚠️ 无有效LLM评估结果，无法针对性重规划")
        return {
            "replan_limit": True,
            "replan_count": replan_count,
            "messages": existing_messages + [
                AIMessage(content="重规划失败：未获取到有效的评估结果")
            ]
        }

    # 3. 重规划次数上限控制（修复配置读取方式，避免硬编码）
    MAX_REPLAN_COUNT = getattr(Settings, "MAX_REPLAN_COUNT", 2)  # 优先从Settings读取
    if replan_count >= MAX_REPLAN_COUNT:
        logger.error(f"⚠️ 达到最大重规划次数（{MAX_REPLAN_COUNT}次），终止任务")
        return {
            "replan_limit": True,
            "replan_count": replan_count,
            "messages": existing_messages + [
                AIMessage(
                    content=f"多次重规划失败（已尝试{replan_count}次，上限{MAX_REPLAN_COUNT}次），"
                            "建议检查任务目标或基础配置"
                )
            ]
        }

    try:
        # 4. 生成新计划（修复参数传递，确保与生成器方法匹配）
        generator = ReplanGenerator()
        new_plan = generator.generate_replan(
            query=user_query,
            original_plan=current_plan,
            final_output=final_output,
            evaluation=evaluation
        )

        # 5. 更新状态（修复计划历史格式，确保可追溯）
        replan_count += 1
        plan_history = state.get("plan_history", [])
        # 追加原始计划记录（首次重规划时）
        if len(plan_history) == 0:
            plan_history.append({
                "plan_id": current_plan.id,
                "timestamp": time.time(),
                "is_replan": False,
                "reason": "原始计划"
            })
        # 追加新计划记录
        plan_history.append({
            "plan_id": new_plan.id,
            "timestamp": time.time(),
            "is_replan": True,
            "reason": evaluation.get("reason", "基于LLM评估重规划")
        })

        logger.info(
            f"✅ 重规划成功 | 新计划ID: {new_plan.id} | 步骤数: {len(new_plan.steps)} "
            f"| 重规划次数: {replan_count}/{MAX_REPLAN_COUNT} | 新计划置信度: {new_plan.confidence:.2f}"
        )

        # 6. 构建返回结果（修复消息格式，避免换行符导致的语法错误）
        return {
            "current_plan": new_plan,
            "plan_history": plan_history,
            "replan_limit": False,
            "replan_count": replan_count,
            "output": "",  # 清空旧执行结果
            "messages": existing_messages + [
                AIMessage(
                    content=f"✅ 重规划完成（第{replan_count}次）\n"
                            f"📋 新计划概要：\n"
                            f"- 计划ID：{new_plan.id}\n"
                            f"- 目标：{new_plan.goal}\n"
                            f"- 步骤数：{len(new_plan.steps)}\n"
                            f"- 置信度：{new_plan.confidence:.2f}\n"
                            f"- 重规划原因：{evaluation.get('reason', '未明确')}\n\n"
                            f"🔧 关键调整：{', '.join(evaluation.get('suggested_adjustments', ['无明确调整方向']))}"
                )
            ]
        }

    except Exception as e:
        # 7. 异常处理（修复错误信息格式，避免过长文本）
        error_msg = str(e)[:100]
        if len(str(e)) > 100:
            error_msg += "..."
        logger.exception(f"❌ 重规划过程异常: {str(e)}")
        return {
            "current_plan": current_plan,
            "replan_limit": True,
            "replan_count": replan_count,
            "messages": existing_messages + [
                AIMessage(
                    content=f"❌ 重规划失败\n"
                            f"错误原因：{error_msg}\n"
                            f"当前状态：保留原始计划，建议检查工具配置或评估结果"
                )
            ]
        }