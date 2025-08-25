from langchain_core.tools import tool
import ast
import operator
import math
import re
from typing import Any, Dict, Union, List, Type


class MetricQueryTools:
    """业务指标查询工具集合"""

    @tool("metric_query")
    @staticmethod
    async def query_metric(query: str) -> Union[str, Dict[str, Any]]:
        """自然语言查询业务指标（示例：河南/深圳/全系统客户量，支持时间筛选）"""
        try:
            # 解析地域标识
            is_henan = "河南" in query
            is_shenzhen = "深圳" in query
            is_all_system = any(kw in query for kw in ["全系统", "全国", "总体", "全部"])

            # 默认返回全量数据（若未指定地域）
            if not is_henan and not is_shenzhen and not is_all_system:
                is_henan = True
                is_shenzhen = False
                is_all_system = True

            # 模拟时间参数（实际需对接业务系统）
            current_year = 2025
            current_month = 7
            last_month = 6
            last_year = 2024

            # 解析时间范围
            time_period = "current_month"  # 默认当月
            if "上月" in query or f"{last_month}月" in query or "上个月" in query:
                time_period = "last_month"
            elif "去年" in query or f"{last_year}年" in query:
                time_period = "last_year"
            elif "今年" in query or f"{current_year}年" in query or "当前年" in query:
                time_period = "current_year"

            # 构建模拟数据（实际需从数据库/API获取）
            data = []
            annotation = [
                {"henan_customer_volume": "河南客户量"},
                {"shenzhen_customer_volume": "深圳客户量"},
                {"all_customer_volume": "全系统客户量"},
                {"month": "统计月份"}
            ]

            # 为不同地域和时间周期设置差异化数据
            if time_period == "current_month":
                # 当月数据（2025-07）
                if is_henan:
                    data.append({
                        "henan_customer_volume": "3950000",
                        "month": f"{current_year}-{current_month:02d}"
                    })
                if is_shenzhen:
                    data.append({
                        "shenzhen_customer_volume": "1820000",
                        "month": f"{current_year}-{current_month:02d}"
                    })
                if is_all_system:
                    data.append({
                        "all_customer_volume": "39400000",
                        "month": f"{current_year}-{current_month:02d}"
                    })

            elif time_period == "last_month":
                # 上月数据（2025-06）
                if is_henan:
                    data.append({
                        "henan_customer_volume": "3980000",
                        "month": f"{current_year}-{last_month:02d}"
                    })
                if is_shenzhen:
                    data.append({
                        "shenzhen_customer_volume": "1779000",
                        "month": f"{current_year}-{last_month:02d}"
                    })
                if is_all_system:
                    data.append({
                        "all_customer_volume": "39600000",
                        "month": f"{current_year}-{last_month:02d}"
                    })

            elif time_period == "last_year":
                # 去年数据（2024-12）
                if is_henan:
                    data.append({
                        "henan_customer_volume": "3650000",
                        "month": f"{last_year}-12"
                    })
                if is_shenzhen:
                    data.append({
                        "shenzhen_customer_volume": "1580000",
                        "month": f"{last_year}-12"
                    })
                if is_all_system:
                    data.append({
                        "all_customer_volume": "35200000",
                        "month": f"{last_year}-12"
                    })

            elif time_period == "current_year":
                # 今年累计数据（2025-01至2025-07）
                if is_henan:
                    data.append({
                        "henan_customer_volume": "27150000",
                        "month": f"{current_year}-YTD"
                    })
                if is_shenzhen:
                    data.append({
                        "shenzhen_customer_volume": "12320000",
                        "month": f"{current_year}-YTD"
                    })
                if is_all_system:
                    data.append({
                        "all_customer_volume": "275800000",
                        "month": f"{current_year}-YTD"
                    })

            return {
                "data": data,
                "annotation": annotation
            }
        except Exception as e:
            return {
                "data": [],
                "annotation": f"指标查询失败: {str(e)}"
            }




class CalculatorTools:
    """数学计算工具集合"""

    # 允许的运算符
    _allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
    }

    # 允许的数学函数和常量
    _allowed_functions = {
        'abs': abs,
        'round': round,
        'max': max,
        'min': min,
        'sum': sum,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'log10': math.log10,
        'exp': math.exp,
        'sqrt': math.sqrt,
        'pi': math.pi,
        'e': math.e
    }

    @tool("calculate")
    @staticmethod
    def calculate(expression: str) -> Union[str, Dict[str, Any]]:
        """
        数学计算工具，支持基本算术运算和常见数学函数

        参数:
            expression: 数学表达式字符串，例如："2 + 3 * 4"、"sin(pi/2)"、"log(100, 10)"

        支持的运算符: +, -, *, /, //, **, %
        支持的函数: abs, round, max, min, sum, sin, cos, tan, log, log10, exp, sqrt
        支持的常量: pi (圆周率), e (自然常数)
        """
        try:
            # 清理表达式中的空白字符
            expression = re.sub(r'\s+', '', expression)
            if not expression:
                return {"result": None, "error": "表达式不能为空"}

            # 解析表达式
            tree = ast.parse(expression, mode='eval')

            # 安全计算表达式
            result = CalculatorTools._eval_ast(tree.body)

            return {
                "expression": expression,
                "result": result,
                "annotation": "计算成功"
            }
        except Exception as e:
            return {
                "expression": expression,
                "result": None,
                "annotation": f"计算失败: {str(e)}"
            }

    @staticmethod
    def _eval_ast(node):
        """递归计算AST节点，只允许安全的操作和函数"""
        if isinstance(node, ast.Expression):
            return CalculatorTools._eval_ast(node.body)

        elif isinstance(node, ast.Num):
            return node.n

        elif isinstance(node, ast.BinOp):
            left = CalculatorTools._eval_ast(node.left)
            right = CalculatorTools._eval_ast(node.right)
            op_type = type(node.op)

            if op_type not in CalculatorTools._allowed_operators:
                raise ValueError(f"不支持的运算符: {node.op}")

            return CalculatorTools._allowed_operators[op_type](left, right)

        elif isinstance(node, ast.UnaryOp):
            operand = CalculatorTools._eval_ast(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            elif isinstance(node.op, ast.UAdd):
                return +operand
            else:
                raise ValueError(f"不支持的单目运算符: {node.op}")

        elif isinstance(node, ast.Name):
            if node.id not in CalculatorTools._allowed_functions:
                raise ValueError(f"不支持的变量或函数: {node.id}")
            return CalculatorTools._allowed_functions[node.id]

        elif isinstance(node, ast.Call):
            func = CalculatorTools._eval_ast(node.func)
            args = [CalculatorTools._eval_ast(arg) for arg in node.args]

            if not callable(func):
                raise ValueError(f"{func} 不是一个函数")

            return func(*args)

        else:
            raise ValueError(f"不支持的表达式元素: {type(node)}")


def get_all_tools():
    """获取所有可用工具（可按需启用/注释）"""
    return [
        MetricQueryTools.query_metric,
        CalculatorTools.calculate  # 启用数学计算工具
    ]


def get_tools_map():
    """工具名称映射（用于快速查找）"""
    return {tool.name: tool for tool in get_all_tools()}
