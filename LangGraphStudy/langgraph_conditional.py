"""
LangGraph 条件分支与循环示例
=============================
本示例演示 LangGraph 的三个核心概念：
1. State：全局状态，贯穿整个图的执行
2. Node：节点，执行具体计算
3. Edge：边，包括无条件边和条件边

工作流：从 count=0 开始，每次递增 1，直到 count >= 3 时停止。
"""

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. 定义状态
# ============================================================
# State 是整个图的"共享内存"。所有节点都读取它、写入它。
class State(TypedDict):
    text: str       # 初始文本
    count: int      # 计数器


# ============================================================
# 2. 定义节点
# ============================================================
# 节点就是一个普通函数：接收 State，返回要更新的字段。
# 注意：只返回需要更新的字段，不需要返回完整的 State。

def increment(state: State) -> dict:
    """将 count 加 1。"""
    new_count = state["count"] + 1
    print(f"  [increment] count: {state['count']} -> {new_count}")
    # 返回更新的state字段，下面实则是告诉langgraph：count 更新为 1
    return {"count": new_count}


# ============================================================
# 3. 定义条件路由函数
# ============================================================
# 路由函数根据当前 State 返回下一个节点的名称。

def should_continue(state: State) -> str:
    """
    如果 count 小于 3，继续循环；否则结束。
    返回值必须是已注册的节点名，或 END。
    """
    if state["count"] < 3:
        return "increment"   # 回到 increment 节点，形成循环
    return END               # 结束


# ============================================================
# 4. 构建图
# ============================================================
workflow = StateGraph(State)

# 添加节点
workflow.add_node("increment", increment)

# 设置入口：START 是 LangGraph 的内置常量，代表图的起点
workflow.add_edge(START, "increment")

# 添加条件边：根据 should_continue 的返回值决定下一步
# 第二个参数是路由函数，第三个参数是路由函数返回值到节点名的映射
workflow.add_conditional_edges(
    "increment",        # 从哪个节点出发
    should_continue,    # 路由函数
    {
        "increment": "increment",  # 返回 "increment" 时回到自己
        END: END,                  # 返回 END 时结束
    },
)

# 编译图（必须调用，否则不能执行）
app = workflow.compile()


# ============================================================
# 5. 运行
# ============================================================
if __name__ == "__main__":
    print("开始执行 LangGraph 条件循环示例：\n")

    result = app.invoke({
        "text": "hello",
        "count": 0,
    })

    print(f"\n最终状态: {result}")
    # 输出: {'text': 'hello', 'count': 3}