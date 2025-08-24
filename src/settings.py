import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Settings:
    """应用配置管理"""
    # OpenAI 相关配置
    OPENAI_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Milvus 相关配置（示例，可扩展）
    MILVUS_HOST = os.getenv("MILVUS_HOST")
    MILVUS_PORT = os.getenv("MILVUS_PORT")

    # 工具配置
    AGENT_CONFIG_PATH = "tool_config.json"  # 工具配置文件路径

    # LLM 模型配置
    LLM_MODEL = "qwen-plus"  # 模型名称，可根据实际调整
    TEMPERATURE = 0.1  # 采样温度
    MAX_TOKENS = 1000  # 最大生成 tokens

    # Agent 执行配置
    MAX_ITERATIONS = 10  # 最大迭代次数
    VERBOSE = True  # 详细日志开关

    # 缓存配置
    CACHE_TTL = 3600  # 缓存过期时间（秒，1小时）
    MAX_CACHE_SIZE = 1000  # 最大缓存条目数

    # 重规划限制
    MAX_AGENT_ITERATIONS = 20  # 智能体重规划最大次数

    MAX_REPLAN_COUNT = 2