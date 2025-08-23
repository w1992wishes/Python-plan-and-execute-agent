from state import AgentState, Plan, PlanStep, PlanType
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
import json
import re
import time
from logger_config import logger
from prompt_setting import get_replanning_system_prompt
from settings import Settings
from agent_tools import get_all_tools, get_tools_map
from langchain_openai import ChatOpenAI


def _parse_replan(plan_text: str, original_plan: Plan, available_tools: list) -> Plan:
    """解析LLM返回的重新规划文本"""
    try:
        # 提取JSON片段（假设LLM按格式返回）
        json_part = re.split(r"<\/think>", plan_text)[-1].strip()
        plan_data = json.loads(json_part)

        # 验证核心字段完整性
        required_fields = ["id", "query", "goal", "plan_type", "steps"]
        for field in required_fields:
            if field not in plan_data:
                raise ValueError(f"重新规划数据缺少必要字段: {field}")

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
                logger.warning(f"重新规划中存在不可用工具: {step.tool}，已清空")
                step.tool = ""  # 清空无效工具

            steps.append(step)

        # 构建完整计划（继承原始计划的核心属性）
        return Plan(
            id=plan_data.get("id", f"replan_{int(time.time())}"),
            query=plan_data.get("query", original_plan.query),
            goal=plan_data.get("goal", original_plan.goal),
            plan_type=PlanType(plan_data.get("plan_type", original_plan.plan_type.value)),
            steps=steps,
            estimated_duration=plan_data.get("estimated_duration", original_plan.estimated_duration),
            confidence=plan_data.get("confidence", original_plan.confidence * 0.9),  # 置信度衰减
            metadata={
                "replan_reason": plan_data.get("replan_reason", "自动调整"),
                "based_on_execution": True
            },
            created_at=time.time()
        )

    except Exception as e:
        logger.error(f"重新规划解析失败: {e}", exc_info=True)
        raise


def _create_replanning_prompt(
        query: str,
        original_plan: Plan,
        executed_steps: list,
        step_results: dict,
        replan_analysis: dict,
        tools_str: str
) -> str:
    """构建重新规划的Prompt（含执行上下文和约束）"""
    # 格式化已执行步骤（带截断结果）
    executed_steps_str = []
    for step_id, result in step_results.items():
        step = next((s for s in original_plan.steps if s.id == step_id), None)
        if step:
            result_str = str(result)
            truncated = result_str[:200] + ("..." if len(result_str) > 200 else "")
            executed_steps_str.append(
                f"步骤 {step.id}: {step.description}\n"
                f" - 工具: {step.tool}\n"
                f" - 结果: {truncated}"
            )

    # 格式化未执行步骤（带依赖）
    remaining_steps = [s for s in original_plan.steps if s.id not in step_results]
    remaining_steps_str = []
    for step in remaining_steps:
        remaining_steps_str.append(
            f"步骤 {step.id}: {step.description}\n"
            f" - 工具: {step.tool}\n"
            f" - 依赖: {', '.join(step.dependencies) if step.dependencies else '无'}"
        )

    # 问题分析模块
    need_replan = replan_analysis.get("need_replan", False)
    reason = replan_analysis.get("reason", "未知")
    issues = replan_analysis.get("issues", [])
    suggestions = replan_analysis.get("suggested_adjustments", [])

    return f"""### 任务重规划需求
根据**执行结果**和**问题分析**，生成**调整后的执行计划**，需满足：  
1. 修复已发现的问题（如步骤失败、结果异常）  
2. 保持与原始目标一致，修正步骤依赖关系  
3. 标注**重规划原因**，评估新计划置信度（可略低于原计划）  


### 原始任务上下文
#### 原始查询  
{query}  

#### 原始计划概要  
- 目标: {original_plan.goal}  
- 类型: {original_plan.plan_type.value}  
- 总步骤: {len(original_plan.steps)}  
- 置信度: {original_plan.confidence:.2f}  


### 执行状态反馈  
#### 已执行步骤（带结果）  
{('\n\n'.join(executed_steps_str)) if executed_steps_str else '无'}  

#### 未执行步骤（带依赖）  
{('\n\n'.join(remaining_steps_str)) if remaining_steps_str else '无'}  


### 问题分析  
- 是否需要重规划: {"是" if need_replan else "否"}  
- 重规划原因: {reason}  
- 具体问题: {('\n- ' + '\n- '.join(issues)) if issues else '无'}  
- 建议调整: {('\n- ' + '\n- '.join(suggestions)) if suggestions else '无'}  


### 可用工具（共 {len(tools_str.splitlines())} 个）  
{tools_str}  


### 输出格式（必须严格遵循JSON）  
```json  
{{  
  "id": "唯一计划ID",  
  "query": "用户原始查询",  
  "goal": "计划目标",  
  "plan_type": "sequential/parallel",  
  "replan_reason": "重规划原因",  
  "steps": [  
    {{  
      "id": "step_1",  
      "description": "步骤描述",  
      "tool": "工具名称",  
      "tool_args": "工具入参（支持引用前置步骤结果）",  
      "input_template": "输入模板（如 {{city}} 的天气）",  
      "dependencies": ["依赖步骤ID"],  
      "expected_output": "预期输出",  
      "confidence": 0.8  
    }}  
  ],  
  "estimated_duration": 60,  // 预计耗时（秒）  
  "confidence": 0.8,         // 新计划置信度  
  "created_at": {int(time.time())}  // 时间戳  
}}  
```"""


class ReplanGenerator:
    """重规划生成器（封装LLM调用和结果解析）"""

    def __init__(self):
        self.tools = get_all_tools()  # 加载所有可用工具
        self.tools_str = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])
        self.tools_map = get_tools_map()  # 工具映射表

        # 初始化LLM客户端
        self.model = ChatOpenAI(
            model=Settings.LLM_MODEL,
            temperature=Settings.TEMPERATURE,
            api_key=Settings.OPENAI_API_KEY,
            base_url=Settings.OPENAI_BASE_URL,
        )

    def generate_replan(
            self,
            query: str,
            original_plan: Plan,
            executed_steps: list,
            step_results: dict,
            replan_analysis: dict
    ) -> Plan:
        """生成调整后的执行计划"""
        # 构建Prompt
        prompt = _create_replanning_prompt(
            query=query,
            original_plan=original_plan,
            executed_steps=executed_steps,
            step_results=step_results,
            replan_analysis=replan_analysis,
            tools_str=self.tools_str
        )

        # 调用LLM
        messages = [
            SystemMessage(content=get_replanning_system_prompt()),  # 系统提示（来自prompt_setting）
            HumanMessage(content=prompt)
        ]
        response = self.model.invoke(messages)
        logger.info(f"重规划LLM响应: \n{response.content}")

        # 解析并返回新计划
        return _parse_replan(
            plan_text=response.content,
            original_plan=original_plan,
            available_tools=[t.name for t in self.tools]
        )


def replan_node(state: AgentState) -> dict:
    """LangGraph节点：根据执行结果动态调整任务计划"""
    logger.info("🔄 重规划节点启动，分析执行结果...")

    # 提取当前状态
    replan_count = state.get("replan_count", 0)
    current_plan = state.get("current_plan")
    step_results = state.get("step_results", {})
    executed_steps = list(step_results.keys())
    replan_analysis = state.get("replan_analysis", {})

    # 重规划次数上限校验（示例：允许1次重规划，可配置）
    if replan_count >= 1:
        logger.error("⚠️ 达到最大重规划次数，终止任务调整")
        return {
            "replan_limit": True,
            "replan_count": replan_count,
            "messages": state.get("messages", []) + [
                AIMessage(content="多次重规划失败，无法继续执行任务")
            ]
        }

    try:
        # 生成新计划
        generator = ReplanGenerator()
        new_plan = generator.generate_replan(
            query=state["input"],
            original_plan=current_plan,
            executed_steps=executed_steps,
            step_results=step_results,
            replan_analysis=replan_analysis
        )

        # 更新状态
        replan_count += 1
        logger.info(
            f"✅ 重规划成功 | 新计划ID: {new_plan.id} | 步骤数: {len(new_plan.steps)} | 重规划次数: {replan_count}"
        )
        print(f"Generated Replan:\n{new_plan}")

        return {
            "current_plan": new_plan,
            "plan_history": state.get("plan_history", []) + [new_plan],
            "replan_limit": False,
            "replan_count": replan_count,
            "messages": state.get("messages", []) + [
                AIMessage(content=f"已根据执行结果调整计划：\n{new_plan}")
            ]
        }

    except Exception as e:
        # 异常处理：记录日志，保留当前计划
        logger.exception(f"❌ 重规划失败: {str(e)}")
        return {
            "current_plan": current_plan,
            "replan_limit": True,
            "replan_count": replan_count,
            "messages": state.get("messages", []) + [
                AIMessage(content=f"重规划异常：{str(e)}，保留当前计划重试")
            ]
        }