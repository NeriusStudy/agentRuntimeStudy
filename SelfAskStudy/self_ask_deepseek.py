"""
Self-Ask 完整实现（使用 DeepSeek API）
========================================
本示例用纯 Python + OpenAI SDK 实现 Self-Ask 范式，
不依赖 LangChain，也不调用任何外部搜索工具。
用本地字典模拟"搜索"，让你专注于理解 Self-Ask 的核心循环。

Self-Ask 的核心机制：
    1. 模型先判断"是否需要提出子问题"
    2. 如果需要，模型输出 Follow up: <子问题>
    3. 系统拦截子问题，调用工具返回答案
    4. 系统把 "Intermediate answer: <答案>" 拼回上下文
    5. 模型看到中间答案后，继续提出下一个子问题或给出最终答案
    6. 最终答案的格式为：So the final answer is: <答案>

与 ReAct 的核心区别：
    ReAct 每步输出 Thought + Action + Action Input；
    Self-Ask 每步输出 Follow up，等待 Intermediate answer。
    Self-Ask 更强调"子问题分解 -> 子问题回答 -> 综合"的问答结构。
"""

import os
import re
import sys
from openai import OpenAI


# ============================================================
# 0. 环境检查 + 客户端初始化
# ============================================================
api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    print("错误：请先设置环境变量 DEEPSEEK_API_KEY")
    print("Linux/macOS:  export DEEPSEEK_API_KEY='sk-你的Key'")
    print("Windows CMD:  set DEEPSEEK_API_KEY=sk-你的Key")
    sys.exit(1)

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",  # DeepSeek 端点（兼容 OpenAI SDK）
)

MODEL = "deepseek-chat"
MAX_STEPS = 8   # 最多允许提出的子问题数量，防止无限循环


# ============================================================
# 1. 模拟知识库（替代外部搜索引擎）
# ============================================================
# 真实场景中，这里应该调用搜索引擎、向量数据库或企业知识库。
# 本示例为了不依赖任何外部工具，用一个本地字典模拟"搜索"。
#
# 匹配逻辑：把子问题去掉标点后，检查是否包含某个关键词。
# 命中第一个就返回对应答案。
#
# 想测试新问题？只需往 FACTS 中加条目即可。

FACTS = {
    # --- 贾斯汀·比伯示例 ---
    "贾斯汀比伯": "贾斯汀·比伯出生于 1994 年。",
    "1994年大师赛": "1994 年大师赛冠军是何塞·马里亚·奥拉萨巴尔。",
    "大师赛": "1994 年大师赛冠军是何塞·马里亚·奥拉萨巴尔。",

    # --- 电影导演示例 ---
    "盗梦空间": "《盗梦空间》的导演是克里斯托弗·诺兰。",
    "诺兰": "克里斯托弗·诺兰是英国人。",
    "皇家赌场": "《007：大战皇家赌场》的导演是马丁·坎贝尔。",
    "坎贝尔": "马丁·坎贝尔是新西兰人。",

    # --- 华盛顿外祖父示例 ---
    "华盛顿的母亲": "乔治·华盛顿的母亲是玛丽·鲍尔·华盛顿。",
    "玛丽鲍尔": "玛丽·鲍尔·华盛顿的父亲是约瑟夫·鲍尔。",

    # --- 登月示例 ---
    "第一次登月": "人类第一次登月是在 1969 年 7 月 20 日。",
    "1969年美国总统": "1969 年美国总统是理查德·尼克松。",
}


def normalize(text: str) -> str:
    """去掉标点和空格，便于关键词匹配。"""
    return re.sub(r"[\s，。？,?！!、：:；;'\"《》()（）·]", "", text)


def search_knowledge_base(question: str) -> str:
    """
    模拟搜索工具：根据子问题返回答案。
    真实场景请替换为搜索引擎/数据库/API 调用。
    """
    q = normalize(question)
    for keyword, answer in FACTS.items():
        if normalize(keyword) in q:
            return answer
    return f"抱歉，知识库中未找到关于「{question}」的信息。"


# ============================================================
# 2. Self-Ask 提示词
# ============================================================
# 关键点：提示词中必须包含 few-shot 示例，教模型输出格式。
#
# Self-Ask 依赖四个结构化标记：
#   Are follow up questions needed here:  触发自问自答
#   Follow up:                            提出子问题
#   Intermediate answer:                  子问题答案（本示例由工具填入）
#   So the final answer is:               最终答案
#
# 模板中的两个占位符：
#   {question}   —— 用户原始问题
#   {scratchpad} —— 累积的"Follow up + Intermediate answer"历史

SELF_ASK_PROMPT = """你是一个善于分解复杂问题的助手。
请使用 Self-Ask 格式回答用户的问题：如果问题需要多步推理，
请依次提出子问题（Follow up），获取中间答案（Intermediate answer），
最后给出最终答案（So the final answer is）。

示例 1：

Question: 穆罕默德·阿里和艾伦·图灵谁活得更久？
Are follow up questions needed here: Yes.
Follow up: 穆罕默德·阿里去世时多少岁？
Intermediate answer: 穆罕默德·阿里去世时 74 岁。
Follow up: 艾伦·图灵去世时多少岁？
Intermediate answer: 艾伦·图灵去世时 41 岁。
So the final answer is: 穆罕默德·阿里

示例 2：

Question: 克雷格列表的创始人出生于哪一年？
Are follow up questions needed here: Yes.
Follow up: 克雷格列表的创始人是谁？
Intermediate answer: 克雷格列表的创始人是克雷格·纽马克。
Follow up: 克雷格·纽马克出生于哪一年？
Intermediate answer: 克雷格·纽马克出生于 1952 年 12 月 6 日。
So the final answer is: 1952 年 12 月 6 日

示例 3：

Question: 乔治·华盛顿的外祖父是谁？
Are follow up questions needed here: Yes.
Follow up: 乔治·华盛顿的母亲是谁？
Intermediate answer: 乔治·华盛顿的母亲是玛丽·鲍尔·华盛顿。
Follow up: 玛丽·鲍尔·华盛顿的父亲是谁？
Intermediate answer: 玛丽·鲍尔·华盛顿的父亲是约瑟夫·鲍尔。
So the final answer is: 约瑟夫·鲍尔

现在请回答下面的问题。重要规则：
- 每一步只输出一个 Follow up，或者直接给出 So the final answer is
- 不要自己编造 Intermediate answer，系统会自动填入工具返回的真实答案
- 如果问题很简单，不需要分解，直接输出 So the final answer is: <答案>

Question: {question}
Are follow up questions needed here: {scratchpad}"""


# ============================================================
# 3. Self-Ask 主循环
# ============================================================
def run_self_ask(question: str, verbose: bool = True) -> str:
    """
    Self-Ask 主循环：
        1. 构造 prompt（含 scratchpad 历史）
        2. 调用 LLM
        3. 解析输出：
           - 若包含 "So the final answer is:" -> 返回答案，结束
           - 若包含 "Follow up:" -> 提取子问题，调用工具获取答案，
             把 "Follow up + Intermediate answer" 追加到 scratchpad
           - 都没有 -> 把错误反馈给模型重试
        4. 循环直到得到最终答案或超过最大步数
    """
    scratchpad = ""  # 累积每一轮的 Follow up + Intermediate answer

    if verbose:
        print(f"\n{'=' * 72}")
        print(f"问题：{question}")
        print(f"{'=' * 72}")

    for step in range(1, MAX_STEPS + 1):
        # ---- 1) 构造 prompt ----
        prompt = SELF_ASK_PROMPT.format(question=question, scratchpad=scratchpad)

        # ---- 2) 调用 DeepSeek ----
        # 关键：用 stop 参数让模型生成到 "\nIntermediate answer:" 就停下，
        # 强制它只输出 Follow up 部分，不允许自己编造中间答案。
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                stop=[
                    "\nIntermediate answer:",
                    "\nIntermediate answer：",
                ],
            )
        except Exception as e:
            print(f"调用 DeepSeek API 失败：{e}")
            return ""

        output = response.choices[0].message.content.strip()

        if verbose:
            print(f"\n--- 第 {step} 步 ---")
            print(output)

        # ---- 3) 检查是否已有最终答案 ----
        final_match = re.search(
            r"So the final answer is\s*[:：]\s*(.+)",
            output,
            re.DOTALL,   # 让 . 匹配换行，以便捕获多行答案
        )
        if final_match:
            answer = final_match.group(1).strip().strip("。.")
            if verbose:
                print(f"\n{'=' * 72}")
                print(f"最终答案：{answer}")
                print(f"{'=' * 72}")
            return answer

        # ---- 4) 检查是否有 Follow up ----
        followup_match = re.search(r"Follow up\s*[:：]\s*([^\n]+)", output)
        if followup_match:
            sub_question = followup_match.group(1).strip()

            # 调用工具获取中间答案（本示例中是模拟知识库）
            sub_answer = search_knowledge_base(sub_question)

            if verbose:
                print(f"[检索工具] 子问题：{sub_question}")
                print(f"[检索结果] {sub_answer}")

            # 把模型本轮输出 + Intermediate answer 追加到 scratchpad
            # 下一轮拼进 prompt 时，模型就能看到"我已经提过什么问题、
            # 得到了什么答案"
            scratchpad += output + f"\nIntermediate answer: {sub_answer}\n"
            continue

        # ---- 5) 输出格式错误，反馈给模型重试 ----
        if verbose:
            print("[解析失败] 模型输出既没有 Follow up 也没有最终答案，要求重试。")

        scratchpad += (
            output
            + "\nIntermediate answer: 输出格式错误，请使用 "
              "`Follow up: <子问题>` 或 `So the final answer is: <答案>` 格式。\n"
        )

    return "已达最大步数，未能完成推理。"


# ============================================================
# 4. 主程序
# ============================================================
if __name__ == "__main__":
    # 示例 1：典型的两跳事实问题
    run_self_ask("贾斯汀·比伯出生那年的大师赛冠军是谁？")

    # 示例 2：需要比较两个独立子链的结果
    run_self_ask(
        "《盗梦空间》的导演和《007：大战皇家赌场》的导演是同一个国家的人吗？"
    )

    # 示例 3：链式依赖（后一个子问题依赖前一个的答案）
    run_self_ask("乔治·华盛顿的外祖父是谁？")

    # 示例 4：两跳事实
    run_self_ask("人类第一次登月时，美国总统是谁？")