"""
LangGraph + DeepSeek 实现 Plan-and-Execute Agent
================================================
本示例使用 LangGraph 的 StateGraph 构建 Plan-and-Execute 流程。

流程：用户输入 → Planner 生成计划 → Executor 逐项执行 → Re-Planner 判断是否继续
"""

import os
import json
from typing import List, TypedDict, Annotated
from operator import add

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END


# ============================================================
# 1. 初始化 DeepSeek LLM
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
def get_month_revenue(store_name: str) -> str:
    """查询指定门店本月的营业额及环比变化。"""
    return f"{store_name}：本月营业额 21 万，环比下降 18%"

@tool
def get_cost_detail(store_name: str) -> str:
    """查询指定门店本月的各项成本明细。"""
    return f"{store_name}：房租 4.5 万、人工 5.2 万、食材 6.8 万、水电杂费 0.8 万"

@tool
def get_month_traffic(store_name: str) -> str:
    """查询指定门店本月的到店客流量及环比变化。"""
    return f"{store_name}：本月到店客流 6200 人次，环比下降 25%"

@tool
def get_nearby_change(store_name: str) -> str:
    """查询指定门店周边最近的经营环境变化。"""
    return f"{store_name}：300 米内新开了 2 家连锁快餐店，其中一家正在做开业大促"

tools = [get_month_revenue, get_cost_detail, get_month_traffic, get_nearby_change]
tool_map = {t.name: t for t in tools}


# ============================================================
# 3. 定义状态结构
# ============================================================
class AgentState(TypedDict):
    """Agent 的全局状态，在节点之间传递。"""
    input: str                              # 用户原始问题
    plan: List[str]                         # 当前计划（步骤列表）
    past_steps: Annotated[List[tuple], add] # 已完成的步骤及结果
    response: str                           # 最终答案


# ============================================================
# 4. Planner 节点：生成计划
# ============================================================
def planner_node(state: AgentState) -> dict:
    """
    Planner 的职责：接收用户目标，一次性输出有序子任务列表。
    这是 Plan-and-Execute 的"规划阶段"。
    """
    tools_desc = "\n".join([f"- {t.name}: {t.description}" for t in tools])

    prompt = f"""你是一个任务规划专家。请将用户的目标分解为有序的、可执行的步骤列表。

可用工具：
{tools_desc}

用户目标：{state['input']}

请只输出一个 JSON 数组，数组的每个元素是一个步骤的字符串描述。
不要输出任何其他内容。例如：
["第一步的描述", "第二步的描述", "第三步的描述"]

计划："""

    response = llm.invoke(prompt)
    content = response.content.strip()

    # 解析 JSON（兼容模型可能输出的 markdown 代码块）
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    plan = json.loads(content.strip())

    print(f"\n[Planner] 生成了 {len(plan)} 个步骤：")
    for i, step in enumerate(plan, 1):
        print(f"  {i}. {step}")

    return {"plan": plan}


# ============================================================
# 5. Executor 节点：执行当前步骤
# ============================================================
def executor_node(state: AgentState) -> dict:
    """
    Executor 的职责：执行计划中的第一个步骤。
    这里使用一个简化的 Function Calling 模式，
    让模型决定用哪个工具、传什么参数。
    """
    if not state["plan"]:
        return {}

    current_step = state["plan"][0]
    print(f"\n[Executor] 执行步骤：{current_step}")

    # 构造工具 Schema（使用 Function Calling 让 Executor 选择工具）
    tools_schema = [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "store_name": {
                            "type": "string",
                            "description": "门店名称，例如：深圳南山店",
                        }
                    },
                    "required": ["store_name"],
                },
            },
        }
        for t in tools
    ]

    prompt = f"""你正在执行一个门店亏损排查任务。
当前需要完成的步骤是：{current_step}

用户原始目标：{state['input']}

请调用合适的工具来完成这个步骤。"""

    response = llm.bind_tools(tools).invoke(prompt)

    # 执行工具调用
    result_text = ""
    if response.tool_calls:
        for tc in response.tool_calls:
            func = tool_map[tc["name"]]
            result = func.invoke(tc["args"])
            result_text += f"[{tc['name']}] {result}\n"
            print(f"  -> {tc['name']}: {result}")
    else:
        result_text = f"未调用工具，模型直接回复：{response.content}"
        print(f"  -> {result_text}")

    # 返回：移除已完成步骤，追加执行结果
    new_plan = state["plan"][1:]
    return {
        "plan": new_plan,
        "past_steps": [(current_step, result_text)],
    }


# ============================================================
# 6. Re-Planner 节点：判断是否继续
# ============================================================
def replanner_node(state: AgentState) -> dict:
    """
    Re-Planner 的职责：
    每完成一步后，判断是继续执行剩余计划，还是给出最终答案。
    这里做了简化处理：如果计划为空，则生成最终答案。
    """
    if state["plan"]:
        # 还有剩余步骤，继续执行
        return {}

    # 计划已完成，生成最终答案
    print("\n[Re-Planner] 计划已全部完成，生成最终答案...")

    past_text = "\n".join([
        f"步骤：{step}\n结果：{result}"
        for step, result in state["past_steps"]
    ])

    prompt = f"""基于以下已完成步骤的执行结果，给出对用户问题的最终回答。

用户问题：{state['input']}

已完成步骤及结果：
{past_text}

请用中文给出一个清晰、结构化的最终回答。"""

    response = llm.invoke(prompt)
    return {"response": response.content.strip()}


# ============================================================
# 7. 构建 LangGraph 状态图
# ============================================================
def should_continue(state: AgentState) -> str:
    """条件判断：计划是否还有剩余步骤。"""
    if state["plan"]:
        return "executor"
    return "replanner"

# 创建状态图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("replanner", replanner_node)

# 设置入口
workflow.set_entry_point("planner")

# 添加边：Planner → Executor
workflow.add_edge("planner", "executor")

# 添加条件边：Executor → Executor（还有步骤）或 Re-Planner（步骤完成）
workflow.add_conditional_edges(
    "executor",
    should_continue,
    {
        "executor": "executor",
        "replanner": "replanner",
    },
)

# Re-Planner → END
workflow.add_edge("replanner", END)

# 编译
app = workflow.compile()


# ============================================================
# 8. 运行
# ============================================================
if __name__ == "__main__":
    result = app.invoke({
        "input": "深圳南山店这个月亏了，帮我排查一下原因。",
        "plan": [],
        "past_steps": [],
        "response": "",
    })

    print(f"\n{'=' * 60}")
    print(f"最终答案:\n{result['response']}")
    print(f"{'=' * 60}")