import json
from typing import Dict, Any


def get_planning_system_prompt(intent_type: str = "SIMPLE_QUERY") -> str:
    """根据意图类型生成规划系统提示词（适配ReAct执行器）"""
    base_system = """你是专业任务规划专家，需根据用户查询生成结构化执行计划。

核心能力：
- 拆解查询为可执行步骤，明确步骤依赖关系
- 从可用工具中选择合适工具，参数需完整
- 客观评估步骤可行性（0.0-1.0置信度）

基础规范：
1. 输出必须是纯JSON，无任何解释性文本
2. 工具必须从提供的"可用工具"中选择
3. 步骤需包含明确的输入输出描述
4. 支持引用前置步骤结果（格式：{step_id_result}，需与ReAct执行器兼容）
"""

    intent_specs = {
        "SIMPLE_QUERY": """
专项要求：
- 聚焦单一查询目标，步骤不超过3个
- 优先选择直接获取数据的工具（如搜索、数据库查询）
- 确保查询参数完整（时间、维度、过滤条件等）
- 工具参数需符合ReAct执行器格式（键值对结构，支持变量引用）
""",
        "COMPARISON": """
专项要求：
- 包含"数据获取→对比分析"两阶段步骤
- 对比维度需与查询强相关（如数值、趋势、特征）
- 分析步骤需明确评估标准（如差异率、优先级）
- 工具参数需使用ReAct执行器兼容的JSON格式
""",
        "ROOT_CAUSE_ANALYSIS": """
专项要求：
- 包含"数据采集→异常定位→根因验证"三阶段
- 当结果存在不确定性时，需生成排查子步骤
- 根因步骤需包含因果关系验证逻辑
- 所有工具调用参数必须可被ReAct执行器直接解析
"""
    }

    return base_system + intent_specs.get(intent_type, intent_specs["SIMPLE_QUERY"])


def create_planning_prompt(
        query: str,
        tools_str: str,
        similar_plans_str: str = "",
        context: Dict[str, Any] = None
) -> str:
    """构建规划阶段的用户提示词（强化工具参数与ReAct执行器对齐）"""
    context_str = json.dumps(context, ensure_ascii=False, indent=2) if context else "{}"

    return f"""### 任务信息
- 用户查询: {query}
- 可用工具: {tools_str}- 上下文: {context_str}- 相似历史计划: {similar_plans_str}

### 计划格式要求（必须严格遵守，适配ReAct执行器）
输出JSON结构如下（字段不可增减，类型需正确）：{{
  "id": "plan_1712345678",  // 格式：plan_时间戳
  "query": "{query}",       // 原样保留用户查询
  "goal": "明确的计划目标",  // 与查询意图一致
  "plan_type": "sequential", // 目前仅支持顺序执行
  "steps": [
    {{
      "id": "step_1",       // 格式：step_序号
      "description": "步骤具体操作描述（需符合ReAct思考逻辑）",
      "tool": "工具名称",    // 必须在可用工具列表中，将被ReAct执行器直接调用
      "tool_args": {{        // 工具参数（必须为JSON对象，ReAct执行器可直接解析）
        "param1": "{query}",
        "param2": "{{step_1_result}}"  // 引用前置步骤结果（ReAct执行器兼容格式）
      }},
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
### 关键适配说明（必须遵守）
1. `tool_args` 必须是标准JSON对象（键值对），确保ReAct执行器可直接作为工具输入
2. 引用前置步骤结果时，使用 `{{step_id_result}}` 格式，与ReAct的变量解析逻辑一致
3. `input_template` 需包含自然语言指令，指导ReAct执行器如何使用工具参数
4. 输出仅包含上述JSON，无其他内容，避免干扰ReAct执行器的解析逻辑
"""
