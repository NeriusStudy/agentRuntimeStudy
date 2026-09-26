"""
DeepSeek ReAct Agent 完整实现
==============================
本示例用纯 Python 实现 ReAct（Reasoning + Acting）范式，不依赖任何 Agent 框架。

核心思想：
    让大模型在"思考 -> 行动 -> 观察"的循环中解决问题。
    模型输出文本格式的 Thought / Action / Action Input，
    由我们的程序解析后执行工具，再把 Observation 拼回上下文，
    然后让模型继续下一轮推理。

与 Function Call 的区别：
    Function Call 让模型输出结构化 JSON 表达调用意图；
    ReAct 让模型输出自然语言文本（Thought + Action），
    调用方用正则解析。二者可以互相替代，也可以结合使用。
"""

import os
import re
import sys
from openai import OpenAI


# ============================================================
# 1. 初始化 DeepSeek 客户端
# ============================================================
api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    print("错误：未设置环境变量 DEEPSEEK_API_KEY")
    print("请先执行：export DEEPSEEK_API_KEY='sk-你的Key'")
    sys.exit(1)

# DeepSeek 完全兼容 OpenAI SDK，只需修改 base_url
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

# 使用的模型：deepseek-chat（通用对话模型，成本低、速度快）
MODEL = "deepseek-chat"


# ============================================================
# 2. 工具定义
# ============================================================
# 工具就是 ReAct 中"Action"可以调用的东西。
# 每个工具都是一个普通的 Python 函数：
#   输入：字符串
#   输出：字符串（会作为 Observation 反馈给模型）

def get_weather(city: str) -> str:
    """
    查询指定城市的当前天气。
    生产环境应该调用真实天气 API（如和风天气、OpenWeatherMap 等），
    这里为了演示使用模拟数据。
    """
    # 清洗输入：去掉模型可能带入的引号、标点、空格
    city = city.strip().strip("'\"").strip("。，,.！!？?")

    weather_data = {
        "北京": "晴，气温 25°C，湿度 40%",
        "上海": "多云，气温 28°C，湿度 65%",
        "深圳": "阵雨，气温 30°C，湿度 80%",
        "杭州": "阴，气温 22°C，湿度 55%",
        "广州": "晴，气温 32°C，湿度 70%",
    }

    if city in weather_data:
        return f"{city}：{weather_data[city]}"

    # 返回错误信息（而不是抛异常），让模型有机会在下一轮自我纠正
    return f"未找到 {city} 的天气数据。当前支持的城市：{', '.join(weather_data.keys())}"


def calculate(expression: str) -> str:
    """
    计算数学表达式。
    生产环境请使用安全的表达式解析库（如 numexpr、sympy、ast.literal_eval 等），
    不要直接使用 eval，存在代码注入风险。
    """
    expression = expression.strip()

    # 白名单校验：只允许数字、运算符、括号、空格和小数点
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "错误：表达式只允许包含数字、+ - * / ( ) 和空格"

    try:
        result = eval(expression)  # 教学演示用；生产环境请替换为安全解析
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


# 工具注册表：工具名 -> {函数, 描述}
# 描述会拼进提示词，帮助模型理解每个工具什么时候该用
TOOLS = {
    "get_weather": {
        "func": get_weather,
        "description": "查询指定城市的当前天气。输入参数为城市名称（如：北京、上海、深圳、杭州、广州）。",
    },
    "calculate": {
        "func": calculate,
        "description": "计算数学表达式。输入参数为合法的数学表达式（如：(25 * 9/5) + 32）。",
    },
}


# ============================================================
# 3. 构造 ReAct 提示词
# ============================================================
# ReAct 的提示词非常关键，它必须清楚地告诉模型：
#   1) 有哪些工具可用
#   2) 输出格式是什么（Thought/Action/Action Input/Final Answer）
#   3) 每条规则是什么（一次只输出一个 Action，不要编造 Observation 等）

def build_tools_description() -> str:
    """把工具注册表格式化成 LLM 可读的文本列表。"""
    lines = []
    for name, info in TOOLS.items():
        lines.append(f"- {name}: {info['description']}")
    return "\n".join(lines)


REACT_PROMPT_TEMPLATE = """你是一个善于使用外部工具解决问题的助手。
你可以调用工具来获取真实数据，而不是凭记忆猜测。

你可以使用以下工具：
{tools}

请严格按照以下格式回答（每次只输出一轮，然后停下来等待系统返回 Observation）：

Thought: 你的推理，说明下一步要做什么
Action: 要使用的工具名称，必须是 [{tool_names}] 之一
Action Input: 传递给工具的输入参数

当你有足够信息回答用户问题时，使用以下格式结束：

Thought: 我已经有足够的信息回答用户了
Final Answer: 对用户问题的最终回答（用中文，简洁准确）

重要规则：
1. 每轮只能输出一个 Action，不要自己编造 Observation，系统会自动把工具执行结果填入
2. Action 必须是工具列表中的名称，不能是其他任何值
3. 如果工具返回错误信息，在下一轮 Thought 中分析原因并调整策略
4. Final Answer 必须直接回答用户的问题，不要包含 Action 或 Action Input 字样

现在开始！

问题: {question}

{history}现在请输出你的下一步："""


def build_prompt(question: str, history: str) -> str:
    """
    构造完整的 ReAct 提示词。
    history 包含之前所有轮次的 Thought / Action / Action Input / Observation。
    """
    return REACT_PROMPT_TEMPLATE.format(
        tools=build_tools_description(),
        tool_names=", ".join(TOOLS.keys()),
        question=question,
        history=history,
    )


# ============================================================
# 4. 解析 LLM 输出
# ============================================================
# 这是 ReAct 最"脆弱"的一环：模型输出的文本必须能被解析出结构。
# 我们用正则从模型输出中提取 Action / Action Input / Final Answer。

def parse_output(text: str) -> dict:
    """
    解析 LLM 的输出，返回以下三种结果之一：

    1) {"type": "final", "answer": "..."}
       模型给出了最终答案，循环应该结束

    2) {"type": "action", "action": "...", "action_input": "..."}
       模型请求调用工具，需要执行

    3) {"type": "error", "message": "..."}
       输出格式不符合要求，需要把错误反馈给模型让它重试
    """
    text = text.strip()

    # ---- 检查 Final Answer ----
    # 用 re.DOTALL 让 . 匹配换行，因为最终答案可能有多行
    fa_match = re.search(r"Final\s*Answer\s*[:：]\s*(.+)", text, re.DOTALL)

    # ---- 检查 Action ----
    # 只捕获到行尾（不用 DOTALL），避免把后续内容也吞进来
    action_match = re.search(r"Action\s*[:：]\s*([^\n]+)", text)

    fa_pos = fa_match.start() if fa_match else -1
    action_pos = action_match.start() if action_match else -1

    # 判断谁先出现：
    #   - 如果 Final Answer 出现在 Action 之前，或者根本没有 Action，就是最终答案
    #   - 否则解析 Action
    if fa_pos != -1 and (action_pos == -1 or fa_pos < action_pos):
        return {"type": "final", "answer": fa_match.group(1).strip()}

    if action_match:
        # 取 Action 后第一个词作为工具名（防止模型多输出内容）
        action_raw = action_match.group(1).strip()
        action = action_raw.split()[0]  # 只取第一个词
        action = action.strip("'\".,;:：，。；")

        # 从 Action 的位置往后查找 Action Input
        rest = text[action_pos:]
        ai_match = re.search(r"Action\s*Input\s*[:：]\s*([^\n]*)", rest)

        if not ai_match:
            return {
                "type": "error",
                "message": "缺少 Action Input，请按照格式补充工具输入参数。",
            }

        action_input = ai_match.group(1).strip()
        # 去掉模型可能加的引号
        action_input = action_input.strip("'\"").strip()

        if not action_input:
            return {
                "type": "error",
                "message": "Action Input 为空，请提供工具所需的输入参数。",
            }

        return {
            "type": "action",
            "action": action,
            "action_input": action_input,
        }

    # 既没有 Action 也没有 Final Answer
    return {
        "type": "error",
        "message": (
            "输出格式不符合要求。请严格按照 Thought/Action/Action Input "
            "或 Thought/Final Answer 的格式输出。"
        ),
    }


# ============================================================
# 5. 执行工具
# ============================================================

def execute_tool(action: str, action_input: str) -> str:
    """
    执行指定工具，返回字符串结果。
    所有异常都被转成字符串反馈给模型，让模型有机会自我纠正。
    """
    if action not in TOOLS:
        return (
            f"错误：工具 '{action}' 不存在。"
            f"可用的工具：{', '.join(TOOLS.keys())}"
        )

    func = TOOLS[action]["func"]
    try:
        result = func(action_input)
        return str(result)
    except Exception as e:
        return f"工具执行出错：{type(e).__name__}: {e}"


# ============================================================
# 6. ReAct 主循环
# ============================================================
# 这是整个 Agent 的心脏：
#   1. 构造 prompt（包含历史轨迹）
#   2. 调用 LLM
#   3. 解析输出
#   4. 如果模型请求工具，执行并追加 Observation 到 history，回到第 1 步
#   5. 如果模型给出 Final Answer，循环结束

def run_react_agent(question: str, max_steps: int = 8, verbose: bool = True) -> str:
    """
    执行 ReAct 循环。
    max_steps 是防止无限循环的安全阀。
    """
    history = ""  # 累积每一轮的 Thought/Action/Action Input/Observation

    if verbose:
        print(f"\n{'=' * 62}")
        print(f"问题: {question}")
        print(f"{'=' * 62}")

    for step in range(1, max_steps + 1):
        if verbose:
            print(f"\n--- 第 {step} 步 ---")

        # ---- 1) 构造提示词 ----
        prompt = build_prompt(question, history)

        # ---- 2) 调用 DeepSeek ----
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,  # ReAct 需要稳定输出，用 0 减少随机性
                # stop 参数：一旦模型开始编造 Observation，立即停止生成
                # 这样我们只需处理 Thought/Action/Action Input 部分
                stop=[
                    "\nObservation:",
                    "\nObservation：",
                    "\n观察:",
                    "\n观察：",
                ],
            )
        except Exception as e:
            print(f"调用 DeepSeek API 失败：{e}")
            return ""

        output = response.choices[0].message.content.strip()

        if verbose:
            # 打印模型本轮输出（Thought + Action + Action Input）
            print(output)

        # ---- 3) 解析输出 ----
        parsed = parse_output(output)

        # ---- 4) 分发处理 ----
        if parsed["type"] == "final":
            # 得到最终答案，循环结束
            if verbose:
                print(f"\n{'=' * 62}")
                print(f"最终答案: {parsed['answer']}")
                print(f"{'=' * 62}")
            return parsed["answer"]

        if parsed["type"] == "action":
            # 执行工具
            action = parsed["action"]
            action_input = parsed["action_input"]

            if verbose:
                print(f"[执行工具] {action}({action_input})")

            observation = execute_tool(action, action_input)

            if verbose:
                print(f"[观察结果] {observation}")

            # 把本轮的 LLM 输出 + Observation 追加到 history，
            # 下一轮拼进 prompt，模型就能看到自己之前做了什么、结果是什么
            history += f"{output}\nObservation: {observation}\n\n"

        else:  # type == "error"
            # 解析失败，把错误作为 Observation 反馈给模型，让它重新输出
            if verbose:
                print(f"[解析错误] {parsed['message']}")

            history += f"{output}\nObservation: {parsed['message']}\n\n"

    # 超过最大步数
    if verbose:
        print(f"\n已达到最大步数（{max_steps}），仍未得到 Final Answer。")
    return "抱歉，我无法在限定步数内完成这个任务。"


# ============================================================
# 7. 主程序
# ============================================================

if __name__ == "__main__":
    # 示例 1：需要多步推理（先查天气，再判断是否需要计算华氏度）
    run_react_agent(
        "杭州现在的天气怎么样？如果温度超过 20 度，帮我算一下华氏度是多少。"
    )

    # 示例 2：纯计算，测试简单工具调用
    run_react_agent("计算 (15 + 27) * 3 - 18 / 2 的结果。")

    # 示例 3：测试错误处理，模型应该根据错误提示调整策略
    run_react_agent("火星的天气怎么样？")