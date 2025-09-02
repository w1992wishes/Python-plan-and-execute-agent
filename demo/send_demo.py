import operator
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.types import Send
from langgraph.graph import END, StateGraph, START
from pydantic import BaseModel, Field
from settings import Settings


import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = Settings.LANGSMITH_API_KEY
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "joke_send_demo"

# 定义模型和提示语 - 已添加JSON相关描述
subjects_prompt = """生成2到5个与主题相关的示例，使用逗号分隔: {topic}。
请以JSON格式返回结果，包含一个名为"subjects"的数组。"""

joke_prompt = """生成一个关于{subject}的笑话。
请以JSON格式返回结果，包含一个名为"joke"的字符串字段。"""

best_joke_prompt = """以下是关于{topic}的一些笑话。选择最好的一个！返回最佳笑话的ID。

{jokes}

请以JSON格式返回结果，包含一个名为"id"的整数字段，表示最佳笑话的索引（从0开始）。"""


class Subjects(BaseModel):
    subjects: list[str]


class Joke(BaseModel):
    joke: str


class BestJoke(BaseModel):
    id: int = Field(description="最佳笑话的索引，从0开始", ge=0)


# 定义模型
model = ChatOpenAI(
    model=Settings.LLM_MODEL,
    temperature=Settings.TEMPERATURE,  # 低温度确保分类稳定
    api_key=Settings.OPENAI_API_KEY,
    base_url=Settings.OPENAI_BASE_URL,
    timeout=20,  # 超时保护
    max_retries=2  # 重试机制提升稳定性
)


# 定义图形的组件
class OverallState(TypedDict):
    topic: str
    subjects: list
    jokes: Annotated[list, operator.add]  # 将所有生成的笑话组合成一个列表
    best_selected_joke: str


class JokeState(TypedDict):
    subject: str


# 生成主题的函数
def generate_topics(state: OverallState):
    prompt = subjects_prompt.format(topic=state["topic"])
    # 使用结构化输出时明确要求JSON格式
    response = model.with_structured_output(Subjects).invoke(prompt)
    return {"subjects": response.subjects}


# 根据主题生成笑话的函数
def generate_joke(state: JokeState):
    prompt = joke_prompt.format(subject=state["subject"])
    response = model.with_structured_output(Joke).invoke(prompt)
    return {"jokes": [response.joke]}


# 定义映射到生成笑话的逻辑
def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]


# 选择最佳笑话的函数
def best_joke(state: OverallState):
    jokes = "\n\n".join([f"笑话 {i}: {joke}" for i, joke in enumerate(state["jokes"])])
    prompt = best_joke_prompt.format(topic=state["topic"], jokes=jokes)
    response = model.with_structured_output(BestJoke).invoke(prompt)
    return {"best_selected_joke": state["jokes"][response.id]}


# 构建图形
graph = StateGraph(OverallState)
graph.add_node("generate_topics", generate_topics)
graph.add_node("generate_joke", generate_joke)
graph.add_node("best_joke", best_joke)
graph.add_edge(START, "generate_topics")
graph.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
graph.add_edge("generate_joke", "best_joke")
graph.add_edge("best_joke", END)
app = graph.compile()

# 可选：绘制图形
try:
    from IPython.display import Image

    Image(app.get_graph().draw_mermaid_png())
except ImportError:
    pass  # 如果没有IPython环境，忽略绘图

# 调用图形以生成笑话列表
for s in app.stream({"topic": "动物"}):
    print(s)
