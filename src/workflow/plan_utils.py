from src.model.plan import Plan, PlanStep, StepTask  # 注意：新增 StepTask 导入
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools.render import render_text_description
from src.log.logger import logger
from src.setting.settings import Settings
from src.workflow.agent_tools import get_all_tools
import time
from typing import Any, List
from src.utils.json_util import extract_json_safely

def validate_plan_for_react(plan: Plan) -> tuple[bool, str]:
    """
    新增：校验 StepTask 合法性（ID唯一性、工具存在性、置信度范围）
    原有：保留 Plan/PlanStep 校验逻辑
    """
    # 1. 原有：校验 Plan 核心字段
    if not plan.id.startswith("plan_"):
        return False, f"计划ID格式错误：{plan.id}（需以'plan_'开头）"
    if not plan.steps:
        return False, "计划不能为空（需至少包含1个步骤）"
    if not (0.0 <= plan.confidence <= 1.0):
        return False, f"计划整体置信度无效：{plan.confidence}（需在0.0-1.0范围内）"

    # 2. 新增：收集全量ID用于唯一性校验（避免 StepTask ID 重复）
    all_step_ids = []
    all_task_ids = []

    # 3. 校验每个 PlanStep 及下属 StepTask
    for step in plan.steps:
        # 3.1 原有：校验 PlanStep
        if not step.id.startswith("step_"):
            return False, f"步骤ID格式错误：{step.id}（需以'step_'开头）"
        if step.id in all_step_ids:
            return False, f"步骤ID重复：{step.id}（所有步骤ID必须唯一）"
        all_step_ids.append(step.id)
        if not (0.0 <= step.confidence <= 1.0):
            return False, f"步骤{step.id}置信度无效：{step.confidence}（需在0.0-1.0范围内）"
        if not step.step_tasks:
            logger.warning(f"步骤{step.id}无并行任务，建议补充 StepTask")  # 警告不阻断，允许空任务

        # 3.2 新增：校验当前 Step 的 StepTask
        for task in step.step_tasks:
            # 校验 Task ID 格式与唯一性
            if not task.id.startswith("step_task_"):
                return False, f"任务ID格式错误：{task.id}（需以'step_task_'开头）"
            if task.id in all_task_ids:
                return False, f"任务ID重复：{task.id}（所有任务ID必须唯一）"
            all_task_ids.append(task.id)

            # 校验 Task 工具合法性（必须在启用工具列表中）
            if task.tool not in get_all_tools():  # 假设 get_all_tools() 返回工具名称列表
                return False, f"任务{task.id}引用无效工具：{task.tool}（不在启用工具列表中）"

            # 校验 Task 置信度与参数
            if not (0.0 <= task.confidence <= 1.0):
                return False, f"任务{task.id}置信度无效：{task.confidence}（需在0.0-1.0范围内）"

    return True, "计划符合ReAct执行器要求（含并行任务校验）"


# ---------------------- 核心修改：计划解析逻辑（新增 StepTask 解析） ----------------------
class BasePlanGenerator:
    def __init__(self):
        self.tools = get_all_tools()
        self.tool_names = [tool.name for tool in self.tools]
        self.tools_str = render_text_description(self.tools)

        self.llm = ChatOpenAI(
            model=Settings.LLM_MODEL,
            temperature=Settings.TEMPERATURE,
            api_key=Settings.OPENAI_API_KEY,
            base_url=Settings.OPENAI_BASE_URL
        )


    def _build_llm_messages(self, system_prompt: str, query: str) -> List[Any]:
        return [
            SystemMessage(content=system_prompt.strip()),
            HumanMessage(content="要拆解的任务是：" + query.strip())
        ]

    def _parse_llm_response(self, query: str, response_content: str):
        """
        核心修改：
        1. 解析 LLM 返回的 step_tasks 数组，生成 StepTask 对象列表
        2. 将 StepTask 列表赋值给 PlanStep 的 step_tasks 字段
        3. 应急计划新增并行任务降级逻辑
        """
        try:
            # 1. 提取JSON数据（原有逻辑不变）
            plan_data = extract_json_safely(response_content)
            if plan_data is None:
                raise ValueError("LLM返回内容无法解析为JSON字典")

            # 2. 解析 PlanStep 列表（新增 StepTask 解析）
            steps = []
            llm_steps = plan_data.get("steps", [])
            for step_idx, step_data in enumerate(llm_steps):
                # 2.1 基础 Step 信息（原有逻辑）
                step_id = step_data.get("id", f"step_{step_idx + 1}_{int(time.time() % 1000)}")
                step_desc = step_data.get("description", f"未命名步骤（{step_idx + 1}）")
                step_confidence = min(max(step_data.get("confidence", 0.7), 0.1), 1.0)

                # 2.2 新增：解析当前 Step 的 StepTask 列表
                step_tasks = []
                llm_tasks = step_data.get("step_tasks", [])  # LLM返回的并行任务数组
                for task_idx, task_data in enumerate(llm_tasks):
                    # 补全 Task 基础信息
                    task_id = task_data.get(
                        "id",
                        f"step_task_{step_idx + 1}_{task_idx + 1}_{int(time.time() % 1000)}"
                    )
                    task_desc = task_data.get("description", f"步骤{step_idx + 1}的并行任务{task_idx + 1}")
                    task_tool = task_data.get("tool", self.tool_names[0] if self.tool_names else "")
                    task_confidence = min(max(task_data.get("confidence", 0.7), 0.1), 1.0)

                    # 过滤无效工具（避免引用不存在的工具）
                    if task_tool not in self.tool_names and self.tool_names:
                        logger.warning(f"任务{task_id}引用无效工具{task_tool}，替换为默认工具{self.tool_names[0]}")
                        task_tool = self.tool_names[0]

                    # 创建 StepTask 对象
                    step_tasks.append(StepTask(
                        id=task_id,
                        description=task_desc,
                        tool=task_tool,
                        confidence=task_confidence,
                    ))

                # 2.3 创建 PlanStep 对象（关联 StepTask 列表）
                steps.append(PlanStep(
                    id=step_id,
                    description=step_desc,
                    step_tasks=step_tasks,  # 核心：将并行任务注入步骤
                    confidence=step_confidence,
                ))

            # 3. 创建 Plan 对象（原有逻辑，补充 estimated_duration）
            plan_id = plan_data.get("id", f"plan_{int(time.time())}")
            plan_goal = plan_data.get("goal", f"处理用户查询：{query[:30]}...")
            plan_confidence = min(max(plan_data.get("confidence", 0.7), 0.1), 1.0)
            plan_duration = max(plan_data.get("estimated_duration", 60.0), 10.0)

            plan = Plan(
                id=plan_id,
                query=plan_data.get("query", query),
                goal=plan_goal,
                steps=steps,
                confidence=plan_confidence,
            )

            # 4. 校验计划合法性（含 StepTask 校验）
            valid, msg = validate_plan_for_react(plan)
            if not valid:
                logger.warning(f"计划兼容性校验未通过：{msg}，将尝试执行")

            return plan

        except Exception as e:
            # 降级逻辑：生成含应急 StepTask 的计划
            error_msg = f"LLM计划解析失败：{str(e)}"
            logger.error(error_msg, exc_info=True)
            return None