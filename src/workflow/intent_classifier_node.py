from src.workflow.state import AgentState  # 导入纯类属性的AgentState
from src.log.logger import logger
from langchain_openai import ChatOpenAI  # 补充LLM依赖（原代码缺失）
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import Dict, Any
from src.setting.settings import Settings  # 补充配置依赖（原代码缺失）
from src.workflow.plan_utils import extract_json_safely  # 补充JSON解析工具（原代码缺失）


class IntentClassifierAgent:
    """完整意图分类Agent（补充LLM调用逻辑，原代码缺失）"""

    def __init__(self):
        """初始化LLM客户端（与全局配置对齐）"""
        self.llm = ChatOpenAI(
            model=Settings.LLM_MODEL,
            temperature=Settings.TEMPERATURE,  # 低温度确保分类稳定
            api_key=Settings.OPENAI_API_KEY,
            base_url=Settings.OPENAI_BASE_URL,
            timeout=20,  # 超时保护
            max_retries=2  # 重试机制提升稳定性
        )

        # 完整提示词模板（明确格式约束，避免解析错误）
        self.system_prompt = """你是专业的指标查询意图分类器，仅负责将用户问题归类到指定类型。
核心规则：
1. 严格按以下3种类型分类，不新增其他类型：
   - SIMPLE_QUERY：指标简单查数（如"2024年1月营收是多少"）
   - COMPARISON：指标对比（如"2024年1月与2月营收差异"）
   - ANALYSIS：指标根因分析（如"为什么2024年1月营收下降"）
2. 必须返回纯JSON格式，无任何前置解释、后置说明或代码块包裹
3. 置信度需客观评估（明确场景0.8+，模糊场景0.5-0.7）
4. 必须包含"intent_type"（英文类型）、"chinese_label"（中文标签）、"confidence"（0-1浮点数）、"reason"（分类依据）"""

        self.user_prompt_template = """### 任务
将用户问题分类到指定意图类型，并返回JSON结果。

### 用户问题
{question}

### 输出格式（必须严格遵守）
{{
  "intent_type": "SIMPLE_QUERY/COMPARISON/ANALYSIS",
  "chinese_label": "指标简单查数/指标对比/指标根因分析",
  "confidence": 0.9,
  "reason": "1-2句话说明分类依据"
}}"""

    def classify(self, question: str) -> Dict[str, Any]:
        """执行意图分类（补充完整LLM调用逻辑，原代码缺失）"""
        try:
            # 1. 格式化提示词
            user_prompt = self.user_prompt_template.format(question=question.strip())

            # 2. 调用LLM获取分类结果
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            response_content = response.content.strip()

            # 3. 安全解析JSON结果（处理LLM可能的格式错误）
            result = extract_json_safely(response_content)
            if "error" in result:
                raise ValueError(f"JSON解析失败：{result['error']}")

            # 4. 校验结果完整性（避免LLM返回缺失字段）
            required_fields = ["intent_type", "chinese_label", "confidence", "reason"]
            missing_fields = [f for f in required_fields if f not in result]
            if missing_fields:
                raise ValueError(f"缺失必要字段：{','.join(missing_fields)}")

            # 5. 校验意图类型有效性（仅允许3种指定类型）
            valid_intent_types = ["SIMPLE_QUERY", "COMPARISON", "ANALYSIS"]
            if result["intent_type"] not in valid_intent_types:
                raise ValueError(f"无效意图类型：{result['intent_type']}，允许值：{valid_intent_types}")

            # 6. 校验置信度范围（0-1浮点数）
            result["confidence"] = round(float(result["confidence"]), 2)
            if not (0.0 <= result["confidence"] <= 1.0):
                raise ValueError(f"置信度超出范围（0-1）：{result['confidence']}")

            # 7. 校验中英文标签一致性（避免LLM返回不匹配结果）
            intent_label_map = {
                "SIMPLE_QUERY": "指标简单查数",
                "COMPARISON": "指标对比",
                "ANALYSIS": "指标根因分析"
            }
            expected_label = intent_label_map[result["intent_type"]]
            if result["chinese_label"] != expected_label:
                logger.warning(
                    f"中英文标签不匹配：{result['intent_type']}→{result['chinese_label']}，已修正为{expected_label}"
                )
                result["chinese_label"] = expected_label

            logger.info(
                f"[意图分类] 成功 | 问题：{question[:30]}... | 类型：{result['intent_type']}（{result['chinese_label']}） | 置信度：{result['confidence']}"
            )
            return result

        except Exception as e:
            error_msg = f"分类逻辑异常：{str(e)}"
            logger.error(f"[意图分类] 失败 | 问题：{question[:30]}... | 原因：{error_msg}", exc_info=True)
            # 降级处理：返回默认分类（确保流程不中断）
            return {
                "intent_type": "SIMPLE_QUERY",
                "chinese_label": "指标简单查数",
                "confidence": 0.5,
                "reason": f"分类异常，启用默认意图（{error_msg}）",
                "error": error_msg  # 标记错误，供后续节点参考
            }


def intent_classifier(state: AgentState):
    """LangGraph意图分类节点（适配纯类AgentState，修复潜在问题）"""
    logger.info(f"[意图分类节点] 启动 | 用户查询：{state.get("input")[:50]}...")

    try:
        # 1. 校验输入有效性（避免空查询）
        user_query = state.get("input")
        if not user_query:
            raise ValueError("用户查询为空，无法分类")

        # 2. 执行意图分类
        classifier = IntentClassifierAgent()
        classify_result = classifier.classify(question=user_query)

        # 3. 更新AgentState（纯类属性操作，修复原代码问题）
        # 修复：原代码调用的set_intent_type方法在AgentState中可能未定义，直接赋值并校验
        valid_intent_types = ["SIMPLE_QUERY", "COMPARISON", "ANALYSIS"]
        if classify_result["intent_type"] in valid_intent_types:
            state.intent_type = classify_result["intent_type"]  # 直接赋值（纯类属性）
        else:
            state.intent_type = "SIMPLE_QUERY"  # 无效类型时降级

        # 其他属性更新（纯类属性直接操作）
        state.intent_info = classify_result  # 存储完整分类结果
        state.need_attention = "error" in classify_result  # 标记是否需要关注
        state.last_error = classify_result.get("error", "")  # 记录错误（若有）

        logger.info(f"[意图分类节点] 完成 | 最终意图类型：{state.intent_type}")
        return {
            "intent_type": state.intent_type,
            "intent_info": state.intent_info,
            "messages": [AIMessage(
                content=f"🔍 意图分类完成！\n- 意图类型：{classify_result['chinese_label']}（{classify_result['intent_type']}）\n- 置信度：{classify_result['confidence']}\n- 分类依据：{classify_result['reason'][:80]}..."
            )],
            "need_attention": state.need_attention,
            "last_error": state.last_error
        }

    except Exception as e:
        error_msg = f"意图分类节点异常：{str(e)}"
        return {
            "intent_type": "SIMPLE_QUERY",
            "messages": [AIMessage(content=f"❌ {error_msg}，已自动降级为「指标简单查数」意图")],
            "need_attention": True,
            "last_error": error_msg
        }


