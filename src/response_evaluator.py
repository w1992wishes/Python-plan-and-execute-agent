from state import AgentState
from logger_config import logger
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from task_planner import Plan  # 假设定义了Plan类
from settings import Settings
import json
from typing import Dict, Any
from json_util import extract_json_safely


class ResponseEvaluator:
    """基于LLM的响应评估器"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=Settings.LLM_MODEL,
            temperature=0.3,  # 低温度确保评估结果稳定
            api_key=Settings.OPENAI_API_KEY,
            base_url=Settings.OPENAI_BASE_URL,
        )

        # 评估系统提示词
        self.system_prompt = """你是一个专业的结果评估专家，负责判断智能体的执行结果是否解决了用户的问题。

评估标准：
1. 相关性：结果是否与用户原始查询相关
2. 完整性：是否完整回答了用户的问题

请根据以下信息进行评估，并严格按照指定格式返回JSON结果：
- 分析结果是否需要重新执行任务（need_replan）
- 给出明确的评估理由（reason）
- 指出存在的问题（issues）
- 提供改进建议（suggested_adjustments）

"""

    def evaluate(self, user_query: str, final_output: str, plan: Plan = None) -> Dict[str, Any]:
        """使用LLM进行评估"""
        # 构建计划信息描述
        plan_info = "无具体执行计划"
        if plan:
            plan_intent = getattr(plan, 'intent_type', '未指定')
            plan_steps = len(plan.steps) if hasattr(plan, 'steps') else 0
            plan_info = f"意图类型: {plan_intent}, 计划步骤数: {plan_steps}"

        # 构建用户提示
        user_prompt = f"""### 评估材料
用户原始查询: {user_query}
执行结果: {final_output}
执行计划信息: {plan_info}

### 输出格式要求
请返回严格的JSON格式，包含以下字段：
{{
  "need_replan": boolean,  // 是否需要重新执行任务
  "reason": string,       // 评估结论的详细理由
  "issues": array,        // 存在的问题列表
  "suggested_adjustments": array  // 改进建议列表
}}

注意：
- need_replan为true表示结果不满足需求，需要重新执行
- 问题和建议要具体，与评估材料直接相关
- 不要添加任何JSON之外的解释文本"""

        # 调用LLM进行评估
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt)
        ]

        try:
            response = self.llm.invoke(messages)
            evaluation = extract_json_safely(response.content.strip())

            # 验证返回格式
            required_fields = ["need_replan", "reason", "issues", "suggested_adjustments"]
            for field in required_fields:
                if field not in evaluation:
                    raise ValueError(f"评估结果缺少必要字段: {field}")

            return evaluation

        except json.JSONDecodeError:
            logger.error(f"LLM返回非JSON格式评估结果: {response.content}")
            return self._default_evaluation("评估结果格式错误")
        except Exception as e:
            logger.error(f"评估过程出错: {str(e)}", exc_info=True)
            return self._default_evaluation(f"评估过程发生错误: {str(e)}")

    def _default_evaluation(self, error_msg: str) -> Dict[str, Any]:
        """评估失败时的默认结果"""
        return {
            "need_replan": True,
            "reason": error_msg,
            "issues": [error_msg],
            "suggested_adjustments": ["检查评估配置或重新执行任务"]
        }


def response_evaluator_node(state: AgentState) -> dict:
    """响应评估节点：使用LLM大模型评估执行结果"""
    logger.info("🔍 LLM评估节点启动，开始智能评估执行结果...")

    # 从状态中提取关键信息
    user_query = state.get("input", "")
    final_output = state.get("output", "")
    current_plan = state.get("current_plan")
    existing_messages = state.get("messages", [])

    # 创建评估器并执行评估
    evaluator = ResponseEvaluator()
    evaluation = evaluator.evaluate(
        user_query=user_query,
        final_output=final_output,
        plan=current_plan
    )

    # 构建评估消息
    eval_message = f"评估结论: {'需要重新执行' if evaluation['need_replan'] else '无需重新执行'}\n"
    eval_message += f"评估理由: {evaluation['reason']}\n"
    logger.info(f"评估结束，结果如下：{eval_message}")

    if evaluation["issues"]:
        eval_message += f"存在问题: {'; '.join(evaluation['issues'])}\n"
    if evaluation["suggested_adjustments"]:
        eval_message += f"改进建议: {'; '.join(evaluation['suggested_adjustments'])}"

    # 组装返回结果
    return {
        "need_replan": evaluation["need_replan"],
        "messages": existing_messages + [AIMessage(content=eval_message)],
        "evaluation": evaluation,  # 保存完整评估信息
        "output": final_output  # 保留原始输出
    }
