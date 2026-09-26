"""
LangChain + DeepSeek 实现 ReAct Agent
=====================================
本示例使用 LangChain 的 create_react_agent + AgentExecutor 实现 ReAct 范式，
底层调用 DeepSeek API。

与手写实现的区别：
    - 手写实现：自己写提示词、自己写正则解析、自己维护循环
    - LangChain：框架帮你完成上述所有工作，你只需定义工具和提示词模板

LangChain 的 AgentExecutor 内部做了哪些事：
    1. 把用户输入 + 工具描述 + 历史轨迹拼成完整 prompt
    2. 调用 LLM 获取输出
    3. 解析输出中的 Action / Action Input / Final Answer
    4. 执行工具，把结果作为 Observation 追加到历史
    5. 循环第 1-4 步，直到解析出 Final Answer 或达到 max_iterations
"""

import os
import sys

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool


# ============================================================
# 0. 环境检查
# ============================================================
if not os.environ.get("DEEPSEEK_API_KEY"):
    print("错误：请先设置环境变量 DEEPSEEK_API_KEY")
    print("Linux/macOS:  export DEEPSEEK_API_KEY='sk-你的Key'")
    print("Windows CMD:  set DEEPSEEK_API_KEY=sk-你的Key")
    sys.exit(1)


# ============================================================
# 1. 初始化 DeepSeek LLM
# ============================================================
# LangChain 的 ChatOpenAI 类支持任何兼容 OpenAI 协议的 API。
# 只需要修改 base_url 指向 DeepSeek，就能把 LLM 从 GPT 切换到 DeepSeek。
llm = ChatOpenAI(
    model="deepseek-chat",                             # DeepSeek 对话模型
    api_key=os.environ.get("DEEPSEEK_API_KEY"),        # 从环境变量读取
    base_url="https://api.deepseek.com",               # DeepSeek 端点
    temperature=0,                                     # ReAct 需要稳定输出
)


# ============================================================
# 2. 定义工具
# ============================================================
# LangChain 用 @tool 装饰器把普通 Python 函数包装成 Tool 对象。
# 函数的 docstring 会自动成为工具描述（description），
# 模型根据这些描述判断"什么时候该用哪个工具"。

@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。输入参数为城市名称，例如：北京、上海、深圳、杭州、广州。"""
    # 清洗模型可能带入的引号、标点
    city = city.strip().strip("'\"。，,.！!？?")

    weather_data = {
        "北京": "晴，气温 25°C，湿度 40%",
        "上海": "多云，气温 28°C，湿度 65%",
        "深圳": "阵雨，气温 30°C，湿度 80%",
        "杭州": "阴，气温 22°C，湿度 55%",
        "广州": "晴，气温 32°C，湿度 70%",
    }

    if city in weather_data:
        return f"{city}：{weather_data[city]}"

    # 返回错误信息而非抛异常，让模型有机会自我纠正
    return f"未找到 {city} 的天气数据。当前支持的城市：{', '.join(weather_data.keys())}"


@tool
def calculate(expression: str) -> str:
    """计算数学表达式。输入参数为合法的数学表达式，例如：(25 * 9/5) + 32。"""
    expression = expression.strip()

    # 白名单校验，防止代码注入
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "错误：表达式只允许包含数字、+ - * / ( ) 和空格"

    try:
        # 教学演示用 eval；生产环境请使用 numexpr、sympy 等安全解析库
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


# 所有工具的列表，传给 create_react_agent
tools = [get_weather, calculate]


# ============================================================
# 3. 定义 ReAct 提示词模板
# ============================================================
# LangChain 要求提示词模板中必须包含以下占位符：
#   {tools}              —— 工具列表（由框架自动填入）
#   {tool_names}         —— 工具名列表（由框架自动填入）
#   {input}              —— 用户输入
#   {agent_scratchpad}   —— 中间步骤历史（Thought/Action/Observation）
#
# 注意：agent_scratchpad 前的 "Thought:" 后面不要加空格，
# 因为 scratchpad 字符串本身以空格开头。

react_prompt = PromptTemplate.from_template("""你是一个善于使用外部工具解决问题的助手。

你可以使用以下工具：
{tools}

请严格按照以下 ReAct 格式回答问题：

Question: 用户的问题
Thought: 你的推理，说明下一步该做什么
Action: 要使用的工具名，必须是 [{tool_names}] 之一
Action Input: 传递给工具的输入参数
Observation: 工具返回的结果
... (Thought/Action/Action Input/Observation 可以重复多次)
Thought: 我现在知道最终答案了
Final Answer: 对用户问题的最终回答（用中文，简洁准确）

重要规则：
1. 每轮只能输出一个 Action，不要一次输出多个
2. Action 必须是工具列表中的名称，不能是其他任何值
3. 不要自己编造 Observation，系统会自动填入工具执行结果
4. 如果工具返回错误信息，请在下一轮 Thought 中分析并调整策略
5. 当有足够信息时，用 Final Answer 结束，不要继续调用工具

开始！

Question: {input}
Thought: {agent_scratchpad}""")


# ============================================================
# 4. 创建 ReAct Agent
# ============================================================
# create_react_agent 做的事情：
#   把 LLM、工具、提示词模板组合成一个 Agent 对象。
#   Agent 的职责是"根据当前上下文决定下一步做什么"（策略层）。
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
)


# ============================================================
# 5. 创建 AgentExecutor（Runtime 层）
# ============================================================
# AgentExecutor 是真正的"运行时"，负责驱动 ReAct 循环。
# 它的参数直接决定了 Agent 的行为和容错性。

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,                   # 打印完整的 ReAct 循环轨迹
    handle_parsing_errors=True,     # 解析失败时自动把错误反馈给模型重试
    max_iterations=10,              # 最大循环轮数，防止无限循环
    return_intermediate_steps=True, # 返回中间步骤，便于调试
)


# ============================================================
# 6. 运行 Agent
# ============================================================
def run(question: str):
    """运行 Agent 并打印结果。"""
    print(f"\n{'=' * 70}")
    print(f"问题: {question}")
    print(f"{'=' * 70}")

    result = agent_executor.invoke({"input": question})

    print(f"\n{'=' * 70}")
    print(f"最终答案: {result['output']}")
    print(f"{'=' * 70}")
    print(f"共执行 {len(result['intermediate_steps'])} 轮工具调用")
    return result


if __name__ == "__main__":
    # 示例 1：多步推理（先查天气，再判断是否需要换算华氏度）
    run("杭州现在的天气怎么样？如果温度超过 25 度，帮我算一下华氏度是多少。")

    # 示例 2：纯计算
    run("帮我计算 (15 + 27) * 3 - 18 / 2 的结果。")

    # 示例 3：错误处理（工具返回错误，模型应调整策略）
    run("火星的天气怎么样？")

    # 示例 4：多次调用工具
    run("对比北京和上海的温度，哪个城市更热？热多少度？")