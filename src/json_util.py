import json
import re

def extract_json_safely(input_str: str):
    """
    安全提取JSON：兼容纯JSON字符串和Markdown JSON代码块，不主动报错
    :param input_str: 输入字符串（可能是纯JSON，也可能是带```json标记的代码块）
    :return: 解析后的JSON对象（dict/list），失败则返回None
    """
    # 第一步：先尝试直接解析（处理纯JSON字符串场景）
    try:
        # 先清理输入前后的空白字符（避免换行/空格导致的解析失败）
        clean_str = input_str.strip()
        if not clean_str:
            return None  # 空字符串直接返回None
        return json.loads(clean_str)
    except json.JSONDecodeError:
        # 直接解析失败，进入第二步：处理Markdown JSON代码块场景
        pass

    # 第二步：提取Markdown代码块中的JSON内容
    try:
        # 正则匹配：提取 ```json 和 ``` 之间的内容（支持多行、前后空白）
        # re.DOTALL 让 . 匹配换行符，\s* 忽略标记与JSON之间的空白
        pattern = r'```json\s*(.*?)\s*```'
        match_result = re.search(pattern, input_str, re.DOTALL)

        if not match_result:
            return None  # 未找到Markdown JSON标记

        # 提取匹配到的JSON原始内容，再次清理空白
        markdown_json = match_result.group(1).strip()
        if not markdown_json:
            return None  # 提取后内容为空

        # 解析清理后的JSON
        return json.loads(markdown_json)
    except (json.JSONDecodeError, Exception):
        # 所有解析尝试失败，返回None（不报错）
        return None