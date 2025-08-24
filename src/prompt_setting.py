import json

def get_planning_system_prompt(intent_type: str = "SIMPLE_QUERY") -> str:
    """根据不同意图类型获取对应的系统提示词（定义角色和核心能力）"""
    base_system = """你是一个高级任务规划专家，擅长根据用户需求设计精准高效的执行计划。

核心能力：
- 理解复杂查询并拆解为可执行步骤
- 合理选择工具并规划步骤依赖关系
- 预测执行风险并设计应对方案
- 参考历史经验优化计划结构

基础规范：
1. 所有计划必须符合可用工具的能力范围
2. 步骤设计需考虑逻辑连贯性
3. 输出必须是严格的JSON格式，不包含任何解释性文本
4. 置信度评估需客观反映步骤可行性"""

    if intent_type == "SIMPLE_QUERY":
        return base_system + """
专项要求：
- 计划应聚焦于单一查询目标，避免冗余步骤
- 优先选择直接获取数据的工具
- 确保查询参数完整（指标、时间、维度等）"""

    elif intent_type == "COMPARISON":
        return base_system + """
专项要求：
- 计划需包含对比数据获取和差异分析两个核心阶段
- 对比维度应明确且与用户需求高度相关
- 分析步骤需包含明确的评估标准"""

    elif intent_type == "ROOT_CAUSE_ANALYSIS":
        return base_system + """
专项要求：
- 计划需包含数据获取→数据风险分析两阶段
- 当分析存在不确定性时，必须生成下级机构/维度的排查步骤
- 根因定位步骤需设计因果关系验证机制"""

    return base_system


def create_planning_prompt(query: str, tools_str: str, similar_plans: list, context: dict = None) -> str:
    """构建用户提示词（提供具体信息和格式约束）"""
    context_str = json.dumps(context, ensure_ascii=False) if context else "无上下文信息"

    # 相似计划格式化
    similar_plans_str = "无相似历史计划"
    if similar_plans:
        similar_plans_str = "\n".join([
            f"- 计划ID: {p.id}, 目标: {p.goal}, 类型: {p.plan_type.value}"
            for p in similar_plans[:3]  # 最多展示3条
        ])

    return f"""### 任务输入信息
- 用户查询: {query}
- 可用工具: {tools_str}
- 上下文信息: {context_str}
- 相似历史计划: {similar_plans_str}

### 计划构建要求
1. 生成分步执行计划，明确步骤间依赖关系
2. 工具必须从可用工具列表中选择，入参支持引用前置步骤结果（如 `step_1_data`）
3. 每个步骤需包含:
   - 清晰的描述和工具选择
   - 完整的输入模板（支持变量插值）
   - 明确的预期输出和置信度（0.0-1.0）
4. 整体计划需标注:
   - 类型（sequential/parallel）
   - 预计耗时（秒）
   - 整体置信度（0.0-1.0）

### 输出格式（必须严格遵循，不可添加额外内容）
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
      "input_template": "输入模板（如 查询深圳本月的客户量）",
      "dependencies": ["step_1"],  # 依赖的步骤ID列表
      "expected_output": "预期输出描述",
      "confidence": 0.8
    }}
  ],
  "estimated_duration": 60,
  "confidence": 0.8,
  "created_at": 1690000000  # 时间戳
}}"""


def get_replanning_system_prompt(intent_type: str = "SIMPLE_QUERY") -> str:
    """获取重规划系统的系统提示"""
    return f"""你是一个高级任务规划专家。你的任务是根据用户查询、可用工具和上下文信息，创建一个结构化的执行计划。

重要指南：
1. 仔细分析用户查询，确定真正的意图和需求
2. 根据意图选择合适的工具和执行策略
3. 创建清晰、可执行的步骤序列，考虑步骤间的依赖关系
4. 为每个步骤指定合适的工具、输入模板和预期输出
5. 评估计划的置信度和预计执行时间
6. 如果查询不明确，创建一个初步计划并标记需要澄清

请严格按照指定的JSON格式返回计划，不要包含任何额外文本。"""