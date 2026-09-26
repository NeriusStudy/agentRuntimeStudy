"""
LangGraph 构建 ReAct Agent 示例
=================================
本示例演示如何用 LangGraph 的 StateGraph 手写一个 ReAct Agent，
包含"思考-行动-观察"循环。

与使用 LangChain 的 create_react_agent 不同，
这里我们从零构建图的每个节点和边，让你彻底理解 Agent 的循环机制。

工作流：
  START -> agent(LLM决策) -> 有工具调用? -> tools(执行工具) -> agent(继续)
                           -> 无工具调用? -> END
"""

import os
import json
from typing_extensions import TypedDict, Annotated
from operator import add

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. 初始化 LLM
# ============================================================
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
)


# ============================================================
# 2. 定义工具
# ============================================================
@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。输入城市名称，例如：北京、上海、深圳。"""
    weather_data = {
        "北京": "晴，气温 25°C，湿度 40%",
        "上海": "多云，气温 28°C，湿度 65%",
        "深圳": "阵雨，气温 30°C，湿度 80%",
    }
    return weather_data.get(city, f"未找到 {city} 的天气数据")

@tool
def calculate(expression: str) -> str:
    """计算数学表达式。输入合法的数学表达式，例如：25 * 9/5 + 32。"""
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "错误：表达式包含不允许的字符"
        return f"{expression} = {eval(expression)}"
    except Exception as e:
        return f"计算错误：{e}"


tools = [get_weather, calculate]
tool_map = {t.name: t for t in tools}

# 将工具绑定到 LLM，使其具备工具调用能力
llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 3. 定义状态
# ============================================================
# 使用 Annotated + add 表示：每次更新 messages 时是"追加"而非"替换"
class AgentState(TypedDict):
    messages: Annotated[list, add]


# ============================================================
# 4. 定义节点
# ============================================================

def agent_node(state: AgentState) -> dict:
    """
    Agent 节点：调用 LLM 进行推理和决策。
    如果 LLM 决定调用工具，它会返回 tool_calls；
    如果 LLM 认为可以回答，它会返回纯文本内容。
    """
    print("\n[Agent] 调用 LLM 进行推理...")
    response = llm_with_tools.invoke(state["messages"])

    if response.tool_calls:
        print(f"[Agent] 决定调用 {len(response.tool_calls)} 个工具")
        for tc in response.tool_calls:
            print(f"  -> {tc['name']}({tc['args']})")
    else:
        print(f"[Agent] 给出最终回答")

    return {"messages": [response]}


def tools_node(state: AgentState) -> dict:
    """
    工具节点：执行 LLM 请求的所有工具调用。
    每个工具的执行结果被包装为 ToolMessage，追加到消息列表中。
    """
    last_message = state["messages"][-1]
    tool_messages = []

    for tc in last_message.tool_calls:
        func = tool_map[tc["name"]]
        result = func.invoke(tc["args"])
        print(f"[Tools] {tc['name']} -> {result}")

        # ToolMessage 需要携带 tool_call_id，与 LLM 的请求对应
        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )

    return {"messages": tool_messages}


# ============================================================
# 5. 定义条件路由
# ============================================================

def should_continue(state: AgentState) -> str:
    """
    判断 LLM 是否请求了工具调用。
    如果请求了，返回 "tools" 去执行工具；
    如果没有，返回 END 结束循环。
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ============================================================
# 6. 构建图
# ============================================================
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tools_node)

# 入口：START -> agent
workflow.add_edge(START, "agent")

# 条件边：agent -> tools 或 END
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)

# 无条件边：tools -> agent（工具执行完后回到 LLM 继续推理）
workflow.add_edge("tools", "agent")

# 编译
app = workflow.compile()


# ============================================================
# 7. 运行
# ============================================================
if __name__ == "__main__":
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("错误：请设置环境变量 DEEPSEEK_API_KEY")
        exit(1)

    question = "北京现在的天气怎么样？如果温度超过 25 度，帮我算一下华氏度。"

    print(f"问题: {question}")
    print("=" * 60)

    result = app.invoke({
        "messages": [
            SystemMessage(content="你是一个乐于助人的助手。请根据工具返回的真实数据回答问题。"),
            HumanMessage(content=question),
        ]
    })

    # 提取最终回答（最后一条消息的内容）
    final_answer = result["messages"][-1].content

    print("\n" + "=" * 60)
    print(f"最终答案: {final_answer}")
    print("=" * 60)