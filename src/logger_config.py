import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # 控制台输出
        logging.FileHandler(filename="agent_workflow.log", encoding="utf-8")  # 日志文件输出
    ]
)

logger = logging.getLogger(__name__)