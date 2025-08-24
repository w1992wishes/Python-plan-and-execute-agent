# Python Plan-and-Execute Agent

## 项目简介

**Python-plan-and-execute-agent** 是一个基于大语言模型（LLM）的多步骤任务规划与执行智能体（Agent）框架。其核心目标是：**针对复杂自然语言查询，自动生成结构化执行计划，并调用外部工具逐步完成任务，支持动态重规划。**

该项目适用于需要多步骤推理、自动化决策、复杂工具链调用等场景，例如智能问答、多源数据分析、自动化运维等。

---

## 核心功能

- **意图识别**：自动识别用户输入的意图类型（如简单查询、对比分析、根因分析）。
- **计划生成**：基于用户问题、上下文和可用工具，自动调用 LLM 生成结构化执行计划。
- **多工具集成**：支持灵活扩展、调用多种外部工具（如指标查询、数学计算等）。
- **计划执行与状态管理**：管理每个计划步骤的执行状态、依赖关系与结果。
- **动态重规划**：在执行过程中根据结果自动分析并调整原有计划，保证任务顺利完成。
- **消息解析与日志**：结构化解析 LLM 交互消息，便于追踪与调试。

---

## 主要模块与架构

```
src/
├── main_agent.py           # Agent 主入口，启动与交互
├── intent_classifier.py    # 意图识别（意图分类Agent）
├── task_planner.py         # 任务规划器（Plan生成、LLM交互）
├── task_replanner.py       # 任务重规划器（根据执行结果调整计划）
├── agent_tools.py          # 工具管理与注册（如指标查询/计算器等）
├── message_parser.py       # LLM消息解析与结构化打印
├── state.py                # 状态与计划（Plan/Step/AgentState等核心结构）
├── response_evaluator.py   # 执行结果分析、是否需要重规划
├── prompt_setting.py       # LLM提示词模板
├── logger_config.py        # 日志配置
└── settings.py             # 全局配置项（如LLM参数、API Key等）
```

### 核心流程

1. **意图识别**：`IntentClassifierAgent` 分析用户问题，输出意图类型。
2. **计划生成**：`PlanGenerator` 基于意图、上下文及工具集，构建 prompt 并调用 LLM 返回结构化计划（Plan）。
3. **计划执行**：`MultiStepAgent` 顺序或并行执行 Plan 步骤，管理依赖与状态。
4. **结果评估**：`response_evaluator` 检查执行结果，判断是否需要重规划。
5. **重规划**：`ReplanGenerator` 根据部分已执行结果与失败情况，自动生成调整后的新计划。
6. **工具扩展**：通过 `agent_tools.py` 注册和管理自定义工具（如 API、数据库查询等）。

---

## 代码片段示例

### 计划生成（PlanGenerator）

```python
class PlanGenerator:
    def generate_plan(self, query: str, intent_type: str, context: dict) -> Plan:
        # 获取历史计划、构建prompt
        prompt = _create_planning_prompt(...)
        # 调用LLM生成计划
        response = self.model.invoke(messages)
        # 解析结构化计划
        return _parse_plan(query, response.content, available_tools)
```

### 动态重规划（ReplanGenerator & response_evaluator）

```python
def _analyze_execution_results(current_plan, executed_steps, step_results):
    # 检查失败步骤，建议是否需要重规划
    if failed_steps:
        analysis["need_replan"] = True
        # ...
    return analysis

class ReplanGenerator:
    def generate_replan(self, ...):
        prompt = _create_replanning_prompt(...)
        response = self.model.invoke(messages)
        # 解析返回的新计划
```

---

## 快速开始

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **配置环境变量**
   - 编辑 `settings.py`，填写所需的 OpenAI API Key、LLM 参数等。

3. **运行主程序**
   ```bash
   python src/main_agent.py
   ```

4. **自定义工具**
   - 在 `agent_tools.py` 中定义新的工具类并注册即可。

---

## 计划与步骤结构

- **Plan**: 包含计划ID、用户查询、目标、类型、步骤列表等
- **PlanStep**: 每一步骤包含唯一ID、描述、调用工具、参数、依赖、置信度、条件等

---

## 技术栈

- Python 3
- [LangChain](https://github.com/langchain-ai/langchain)
- OpenAI GPT/LLM
- 标准库（dataclasses/enum/typing）
