from src.log.logger import logger
import re
import json
from typing import Dict, Any, Optional

def extract_json_safely(input_str: str) -> Optional[Dict[str, Any]]:
    try:
        clean_str = input_str.strip()
        if not clean_str:
            logger.warning("[JSON提取] 输入字符串为空，无法解析")
            return None
        parsed = json.loads(clean_str)
        if not isinstance(parsed, dict):
            logger.warning(f"[JSON提取] 解析结果非字典类型（{type(parsed).__name__}），不符合计划格式")
            return None
        return parsed
    except json.JSONDecodeError as e:
        logger.debug(f"[JSON提取] 直接解析失败（{str(e)[:50]}），尝试提取Markdown代码块")

    try:
        pattern = r'```(?:json)?\s*(.*?)\s*```'
        match_result = re.search(pattern, input_str, re.DOTALL)
        if not match_result:
            logger.warning("[JSON提取] 未找到Markdown代码块标记（如```json）")
            return None

        markdown_json = match_result.group(1).strip()
        if not markdown_json:
            logger.warning("[JSON提取] Markdown代码块内内容为空")
            return None

        parsed = json.loads(markdown_json)
        if not isinstance(parsed, dict):
            logger.warning(f"[JSON提取] 代码块内解析结果非字典类型（{type(parsed).__name__}）")
            return None
        return parsed
    except json.JSONDecodeError as e:
        logger.error(f"[JSON提取] Markdown代码块解析失败：{str(e)}，原始内容：{markdown_json[:100]}...")
    except Exception as e:
        logger.error(f"[JSON提取] 未知异常：{str(e)}", exc_info=True)

    return None