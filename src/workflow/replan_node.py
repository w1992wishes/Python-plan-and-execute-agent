from src.workflow.state import AgentState
from src.model.plan import Plan
from langchain_core.messages import AIMessage
from src.log.logger import logger
from src.workflow.plan_utils import BasePlanGenerator
from src.setting.settings import Settings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Union
import time


class Response(BaseModel):
    """Response to user."""
    response: str = Field(description="直接回复用户的内容，需汇总所有任务结果")


class Act(BaseModel):
    """Action to perform."""
    action: Union[Response, Plan] = Field(
        description="要执行的操作：回复用户用Response；需继续执行用Plan（必须包含PlanStep→StepTask层级）"
    )


class TaskReplanner(BasePlanGenerator):
    async def aupdate_plan(self, state: AgentState):
        """异步重规划（适配三层结构：处理并行任务）"""
        original_plan = state.get("current_plan")
        executed_tasks = state.get("executed_tasks", [])
        user_query = state.get("input")

        # 1. 格式化原计划（展示三层结构：步骤包含并行任务）
        original_plan_steps = []
        for step_idx, step in enumerate(original_plan.steps, 1):
            # 步骤基础信息
            step_info = (
                f"step{step_idx}（ID：{step.id}）："
                f"描述={step.description} | "
                f"并行任务数={len(step.step_tasks)}"
            )
            original_plan_steps.append(step_info)

            # 列出该步骤下的所有并行任务
            for task_idx, task in enumerate(step.step_tasks, 1):
                task_info = (
                    f"任务（描述：{task.description}）："
                    f"工具={task.tool}"
                )
                original_plan_steps.append(task_info)
        formatted_original_plan = "\n".join(original_plan_steps)

        # 2. 格式化已执行任务（按任务名去重，保留成功记录）
        past_tasks = []
        seen_task_descs = set()
        for task in executed_tasks:
            if task.get("status") == "success" and task["description"] not in seen_task_descs:
                seen_task_descs.add(task["description"])
                # 关联任务所属步骤
                task_step = next(
                    (step.id for step in original_plan.steps
                     if any(t.description == task["description"] for t in step.step_tasks)),
                    "未知步骤"
                )
                past_tasks.append(
                    f"任务：{task['description']}（步骤{task_step}）→ "
                    f"结果：{str(task['result'])[:50]}"
                )
        formatted_past_steps = "\n".join(past_tasks) if past_tasks else "无"

        # 3. 重规划提示词（适配三层结构输出）
        replanner_prompt = ChatPromptTemplate.from_template("""
你是专业任务重规划专家，需基于三层结构（Plan→PlanStep→StepTask）优化计划：

核心规则：
1. 保留未执行的StepTask，删除已完成任务；若步骤所有任务已完成，删除整个步骤
2. 同步骤下的StepTask必须无依赖（支持并行执行）
3. 输出必须是纯JSON，无任何解释性文本

用户原始查询：{input}
原计划（含并行任务）：
{plan}
已执行任务（成功记录）：
{past_steps}

输出格式（二选一）：

1. 存在未执行任务（返回含StepTask的Plan）：
{{
  "action": {{
    "id": "",
    "query": "{input}",
    "goal": "",
    "steps": [
      {{
        "id": "step_1",
        "description": "步骤描述（含并行任务目标）",
        "confidence": 0.8-1.0,
        "step_tasks": [
          {{
            "id": "step_task_1",
            "description": "任务具体操作",
            "tool": "工具名",
            "confidence": 0.8-1.0
          }}
        ]
      }}
    ],
    "confidence": 0.8-1.0
  }}
}}

2. 所有任务完成（返回用户回复）：
{{
  "action": {{
    "response": "基于所有任务结果的最终回答"
  }}
}}

注意：
- 步骤级串行执行，任务级并行执行
- 已执行任务结果可直接引用（如"根据任务step_task_1_1的结果..."）
- 工具参数必须是JSON对象
""")

        # 4. 调用LLM
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=Settings.LLM_MODEL,
            temperature=0.3,
            api_key=Settings.OPENAI_API_KEY,
            base_url=Settings.OPENAI_BASE_URL,
        )

        replanner = replanner_prompt | llm.with_structured_output(Act)
        replan_response = await replanner.ainvoke({
            "input": user_query,
            "plan": formatted_original_plan,
            "past_steps": formatted_past_steps
        })

        return replan_response


async def replan(state: AgentState):
    logger.info(f"[重规划节点] 启动 | 原计划ID：{state.get("current_plan").id}")
    replanner = TaskReplanner()

    replan_response = await replanner.aupdate_plan(state)

    # 处理任务完成情况
    if isinstance(replan_response.action, Response):
        # 清空计划（保持三层结构完整性）
        current_plan = Plan(
            id=f"plan_completed_{int(time.time())}",
            query=state.get("input", "用户查询"),
            goal="任务已完成",
            steps=[],
            confidence=1.0
        )
        # 汇总已执行任务
        completed_tasks = [t for t in state.get("executed_tasks") if t.get("status") == "success"]

        logger.info(f"[重规划节点] 任务完成 | 已执行任务数：{len(completed_tasks)}")

        return {
            "current_plan": [],
            "task_completed": True,
            "message": [AIMessage(
                content=f"🎉 任务完成！\n已执行任务：{len(completed_tasks)}个\n"
                        f"最终结果：{replan_response.action.response[:150]}..."
            )]
        }

    # 处理更新后的计划
    else:
        updated_plan: Plan = replan_response.action
        # 验证计划结构合法性
        valid_plan = all(hasattr(step, "step_tasks") for step in updated_plan.steps)
        if not valid_plan:
            raise ValueError("重规划返回的计划不符合三层结构要求")

        # 生成步骤和任务摘要
        step_summaries = []
        for step in updated_plan.steps[:3]:  # 只显示前3个步骤
            task_count = len(step.step_tasks)
            step_summaries.append(
                f"- {step.id}: {step.description[:30]}...（并行任务：{task_count}个）"
            )
        if len(updated_plan.steps) > 3:
            step_summaries.append(f"- ... 还有{len(updated_plan.steps) - 3}个步骤")

        logger.info(f"[重规划节点] 完成 | 剩余步骤：{len(updated_plan.steps)}")
        return {
            "current_plan": updated_plan,
            "plan_history": [updated_plan],
            "messages": [AIMessage(
                content=f"🔄 重规划完成！\n新计划ID：{updated_plan.id}\n"
                        f"剩余步骤：{len(updated_plan.steps)} | 剩余任务：{sum(len(s.step_tasks) for s in updated_plan.steps)}\n"
                        f"步骤摘要：\n" + "\n".join(step_summaries)
            )]
        }
