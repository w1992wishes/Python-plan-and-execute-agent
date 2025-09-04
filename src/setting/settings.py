import os

class Settings:
    """应用配置管理"""
    # OpenAI 相关配置
    OPENAI_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # LLM 模型配置
    LLM_MODEL = "qwen-plus"  # 模型名称，可根据实际调整
    TEMPERATURE = 0.1  # 采样温度
