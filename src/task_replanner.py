from state import AgentState, Plan, PlanStep  # 确保导入PlanStep
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from logger_config import logger
from plan_utils import BasePlanGenerator
from settings import Settings
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field
from typing import List, Union

class Response(BaseModel):
    """Response to user."""
    response: str = Field(description="直接回复用户的内容")

class Act(BaseModel):
    """Action to perform."""
    action: Union[Response, Plan] = Field(
        description="要执行的操作。如果要回复用户，使用Response；如果需要进一步使用工具，使用Plan。"
    )

class TaskReplanner(BasePlanGenerator):
    async def aupdate_plan(self, state: AgentState):
        """异步重规划（核心：补充已执行步骤上下文）"""
        original_plan = state.current_plan
        executed_steps = state.executed_steps

        # 1. 格式化原计划步骤（适配模板的"step编号: 描述"格式）
        original_plan_steps = []
        for idx, step in enumerate(original_plan.steps, 1):
            original_plan_steps.append(
                f"step{idx}: {step.description}（工具：{step.tool}，参数：{step.tool_args}）"
            )
        formatted_original_plan = "\n".join(original_plan_steps)

        # 2. 格式化已执行步骤（适配模板的"任务: 结果"格式）
        past_steps = []
        for item in executed_steps:
            step_id = item["step_id"]
            step = next((s for s in original_plan.steps if s.id == step_id), None)
            step_desc = step.description if step else f"步骤{step_id}"
            # 结果截断避免过长
            result = str(item["result"])[:100] + ("..." if len(str(item["result"])) > 100 else "")
            past_steps.append(f"任务：{step_desc} → 结果：{result}")
        formatted_past_steps = "\n".join(past_steps) if past_steps else "无"

        replanner_prompt = ChatPromptTemplate.from_template(
            """你是专业任务重规划专家，需根据执行进度优化原有计划，确保与ReAct执行器兼容。

            核心规则：
            1. 必须基于原始目标、原计划和已执行步骤结果进行重规划
            2. 仅保留未执行的步骤（删除已完成步骤），新增步骤需补充在未执行步骤之后
            3. 若所有步骤已完成，直接返回用户最终答案
            4. 步骤描述需明确包含工具调用意图（如"使用metric_query查询..."、"使用calculate计算..."）

             Your objective was this:
            {input}

            Your original plan was this (each step is "step编号: 描述"):
            {plan}

            You have currently done these steps (任务: 结果):
            {past_steps}

            You MUST return your response in one of the following JSON formats (escape curly braces correctly):

            输出格式要求（必须严格遵守，否则执行器无法解析）：
            - 若有剩余步骤，返回：
{{
  "id": "plan_1712345678",  // 格式：plan_时间戳
  "query": "用户查询",       // 原样保留用户查询
  "goal": "明确的计划目标",  // 与查询意图一致
  "plan_type": "sequential", // 目前仅支持顺序执行
  "steps": [
    {{
      "id": "step_1",       // 格式：step_序号
      "description": "步骤具体操作描述（需符合ReAct思考逻辑）",
      "tool": "工具名称",    // 必须在可用工具列表中，将被ReAct执行器直接调用
      "tool_args": "工具参数（必须为JSON）"        // 工具参数（必须为JSON对象，ReAct执行器可直接解析）
      "input_template": "自然语言输入模板（供ReAct执行器生成工具调用指令）",
      "dependencies": [],   // 依赖的步骤ID列表（无依赖留空）
      "expected_output": "预期输出的结构化描述（需为ReAct执行器可返回的格式）",
      "confidence": 0.8     // 0.0-1.0的浮点数
    }}
  ],
  "estimated_duration": 60,  // 预计总耗时（秒）
  "confidence": 0.8,         // 整体计划置信度
  "created_at": 1712345678   // 生成时间戳（整数）
}}

            - 若任务已完成，返回：
            {{
              "action": {{
                "response": "直接回复用户的内容（包含最终答案，需基于已执行步骤结果）"
              }}
            }}

            注意：
            - 步骤序号仅用于展示，实际执行时会自动生成不重复的ID
            - 已执行步骤结果可直接引用（如"根据步骤2的结果，计算..."）
            - 工具参数需符合要求（如metric_query的query参数为自然语言，calculate的expression为数学公式）
            """
        )

        # 4. 异步调用LLM
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=Settings.LLM_MODEL,
            temperature=0.3,
            api_key=Settings.OPENAI_API_KEY,
            base_url=Settings.OPENAI_BASE_URL,
        )

        replanner = replanner_prompt | llm.with_structured_output(Act)

        replan_response = await replanner.ainvoke({
            "input": state.input,
            "plan": formatted_original_plan,
            "past_steps": formatted_past_steps if past_steps else "No steps completed yet."
        })

        return replan_response

# 异步重规划节点（保持原有逻辑，适配新的Plan结构）
async def task_replanner_node(state: AgentState) -> AgentState:
    logger.info(f"[重规划节点] 启动 | 原计划ID：{state.current_plan.id}")
    replanner = TaskReplanner()
    replan_response = await replanner.aupdate_plan(state)

    # 根据 replan 结果更新 state
    if isinstance(replan_response.action, Response):
        state.task_completed = True
        final_result = replan_response.action.response
        state.add_message(AIMessage(
            content=f"🎉 任务完成！\n已执行步骤：{len(state.executed_steps)}\n最终结果：{str(final_result)[:150]}..."
        ))
        return state
    else:
        # 若返回新计划，更新 plan 字段（保留剩余步骤）
        # 更新计划状态
        updated_plan = replan_response.action
        state.set_current_plan(updated_plan)
        state.add_message(AIMessage(
            content=f"🔄 重规划完成！\n新计划ID：{updated_plan.id}\n剩余步骤：{len(updated_plan.steps)}\n"
                    f"步骤详情：\n" + "\n".join([f"- {s.id}: {s.description}" for s in updated_plan.steps[:3]])
                    + ("..." if len(updated_plan.steps) > 3 else "")
        ))
        logger.info(f"[重规划节点] 完成 | 剩余步骤：{len(updated_plan.steps)}")
        return state





