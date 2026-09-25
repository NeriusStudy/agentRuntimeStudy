"""
DeepSeek Function Call 完整示例
================================
本示例演示如何使用 DeepSeek API 实现 Function Call（工具调用）。
流程：用户提问 -> 模型决定调用工具 -> 程序执行工具 -> 结果返回模型 -> 模型生成最终回答。
"""

import os
import json
from openai import OpenAI

# ============================================================
# 第 1 步：初始化 DeepSeek 客户端
# ============================================================
# DeepSeek 完全兼容 OpenAI SDK，只需修改 base_url 即可
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),  # 从环境变量读取 API Key
    base_url="https://api.deepseek.com",        # DeepSeek 的 API 端点
)


# ============================================================
# 第 2 步：定义工具（JSON Schema）
# ============================================================
# 这里的 tools 数组告诉模型"有哪些函数可用、每个函数需要什么参数"。
# 格式与 OpenAI 的 tools 参数完全一致。
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前天气。返回温度、天气状况和湿度。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海、深圳",
                    }
                },
                "required": ["city"],  # city 是必填参数
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式。支持加减乘除和括号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，例如：25 * 9/5 + 32",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


# ============================================================
# 第 3 步：实现本地函数
# ============================================================
# 这些是真正被执行的代码。模型不会执行它们，
# 模型只会输出"请调用 get_weather，参数是 city=北京"，
# 真正执行的是下面这些 Python 函数。

def get_weather(city: str) -> str:
    """
    模拟天气查询函数。
    生产环境中，这里应该调用真实的天气 API（如和风天气、OpenWeatherMap 等）。
    """
    # 模拟数据，用于演示
    weather_data = {
        "北京": "晴，气温 25°C，湿度 40%",
        "上海": "多云，气温 28°C，湿度 65%",
        "深圳": "阵雨，气温 30°C，湿度 80%",
        "杭州": "阴，气温 22°C，湿度 55%",
    }
    # 如果城市不在模拟数据中，返回错误提示（让模型有机会纠正）
    return weather_data.get(city, f"未找到 {city} 的天气数据，请检查城市名称是否正确。")


def calculate(expression: str) -> str:
    """
    计算数学表达式。
    生产环境请使用安全的表达式解析库（如 numexpr、sympy），
    不要直接使用 eval，存在代码注入风险。
    """
    try:
        # 限制只允许数字、运算符和括号，防止恶意代码执行
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return f"表达式包含不允许的字符，请只使用数字和 +-*/() 运算符。"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


# 工具名到函数的映射表，用于根据模型返回的函数名动态查找并执行对应的 Python 函数
available_functions = {
    "get_weather": get_weather,
    "calculate": calculate,
}


# ============================================================
# 第 4 步：封装单次 API 调用
# ============================================================
def send_messages(messages):
    """
    向 DeepSeek API 发送消息，返回模型的响应消息。
    每次调用都传入完整的对话历史（messages），让模型知道上下文。
    """
    response = client.chat.completions.create(
        model="deepseek-chat",   # DeepSeek 的对话模型
        messages=messages,
        tools=tools,             # 传入工具定义
        tool_choice="auto",      # 让模型自行决定是否调用工具
    )
    return response.choices[0].message


# ============================================================
# 第 5 步：核心对话循环（Function Call 引擎）
# ============================================================
def run_conversation(user_input: str):
    """
    完整的 Function Call 对话循环。
    逻辑：调用模型 -> 检查 tool_calls -> 执行工具 -> 追加结果 -> 再次调用模型
    循环直到模型不再请求工具调用，返回最终自然语言回答。
    """
    # 初始化对话历史
    messages = [
        {
            "role": "system",
            "content": "你是一个乐于助人的助手。请根据工具返回的真实数据回答问题。"
                       "如果工具返回了错误信息，请尝试分析原因并决定是否重试或更换参数。",
        },
        {"role": "user", "content": user_input},
    ]

    print(f"\n{'='*60}")
    print(f"用户: {user_input}")
    print(f"{'='*60}")

    max_turns = 10  # 最大循环次数，防止无限循环
    turn = 0

    while turn < max_turns:
        turn += 1
        print(f"\n--- 第 {turn} 轮 ---")

        # 调用模型
        message = send_messages(messages)

        # 把模型的响应追加到对话历史中
        # 注意：这一步至关重要。模型下一轮需要看到自己刚才请求了什么。
        messages.append(message)

        # 检查模型是否请求了工具调用
        if not message.tool_calls:
            # 模型没有请求工具调用，说明它已经准备好最终答案
            print(f"\n最终答案: {message.content}")
            return message.content

        # 模型请求了工具调用，逐个执行
        print(f"模型请求调用 {len(message.tool_calls)} 个工具:")

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"  -> 工具: {function_name}")
            print(f"     参数: {function_args}")

            # 检查函数是否存在
            if function_name not in available_functions:
                tool_result = f"错误：函数 '{function_name}' 不存在。可用的函数有：{list(available_functions.keys())}"
            else:
                # 执行实际的 Python 函数
                function_to_call = available_functions[function_name]
                try:
                    tool_result = function_to_call(**function_args)
                except TypeError as e:
                    # 参数不匹配时的错误处理
                    tool_result = f"参数错误：{e}。请检查参数名称和类型是否正确。"
                except Exception as e:
                    # 其他执行错误的处理
                    tool_result = f"工具执行失败：{type(e).__name__}: {e}"

            print(f"     结果: {tool_result}")

            # 把工具执行结果追加到对话历史
            # 关键：必须使用 tool_call.id 作为 tool_call_id，
            # 这样模型才能把结果和刚才的请求对应起来
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(tool_result),
            })

    # 超过最大轮次仍未得到最终答案
    print(f"\n达到最大轮次限制（{max_turns}），强制终止。")
    return "抱歉，处理超时，请尝试简化问题。"


# ============================================================
# 第 6 步：运行示例
# ============================================================
if __name__ == "__main__":
    # 示例 1：需要先查天气，再根据温度计算华氏度
    run_conversation(
        "杭州现在的天气怎么样？如果温度超过 25 度，帮我算一下华氏度是多少。"
    )

    # 示例 2：纯计算
    run_conversation("帮我计算 (15 + 27) * 3 - 18 / 2 的结果。")

    # 示例 3：查询不存在的城市（测试错误处理）
    run_conversation("火星的天气怎么样？")