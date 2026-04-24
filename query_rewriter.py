import json
import re
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from config import COMPLEXITY_SIGNAL_THRESHOLD, COMPLEX_LENGTH_THRESHOLD, MAX_SUB_QUERIES
from topic_router import TOPIC_KEYWORDS

COMPLEX_CONNECTORS = ["同时", "并且", "以及", "还有", "另外", "而且", "一边", "伴有"]
INTENT_MARKERS = ["怎么", "如何", "为什么", "原因", "是否", "会不会", "怎么办", "要不要", "需不需要", "怎么治疗"]

QUERY_REWRITE_PROMPT = PromptTemplate(
    template="""
你是睡眠健康RAG系统的查询重构器。

任务：
1. 将用户的复杂问题拆分成 2 到 4 个更适合检索的子问题。
2. 子问题要覆盖原问题的主要意图，但不要脱离睡眠健康语境。
3. 子问题要简洁、自然、适合直接送入向量检索。
4. 不要回答问题本身，只做检索导向的改写。
5. 输出必须是严格 JSON，格式如下：
{{
  "reason": "一句话说明为何这是复杂问题",
  "sub_queries": ["子问题1", "子问题2", "子问题3"]
}}

要求：
- 最多输出 {max_sub_queries} 个子问题。
- 子问题之间尽量从不同角度覆盖：症状 / 原因 / 风险 / 处理 / 就医建议。
- 如果原问题包含多个症状或多个意图，拆分时要保留这些重点。
- 子问题中不要使用编号前缀。

用户原问题：{question}
""",
    input_variables=["question", "max_sub_queries"],
)


@dataclass
class QueryRewriteResult:
    original_question: str
    rewritten_queries: List[str]
    is_complex: bool
    complexity_method: str
    complexity_reason: str
    rewrite_method: str
    rewrite_reason: str


class QueryRewriter:
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm

    def rewrite(self, question: str) -> QueryRewriteResult:
        is_complex, reason = self._is_complex_question(question)
        if not is_complex:
            return QueryRewriteResult(
                original_question=question,
                rewritten_queries=[question],
                is_complex=False,
                complexity_method="rule",
                complexity_reason=reason,
                rewrite_method="none",
                rewrite_reason="问题较简单，直接使用原问题检索。",
            )

        if self.llm is not None:
            llm_result = self._rewrite_with_llm(question)
            if llm_result is not None:
                return QueryRewriteResult(
                    original_question=question,
                    rewritten_queries=llm_result["sub_queries"],
                    is_complex=True,
                    complexity_method="rule",
                    complexity_reason=reason,
                    rewrite_method="llm",
                    rewrite_reason=llm_result["reason"],
                )

        return QueryRewriteResult(
            original_question=question,
            rewritten_queries=[question],
            is_complex=True,
            complexity_method="rule",
            complexity_reason=reason,
            rewrite_method="fallback",
            rewrite_reason="已识别为复杂问题，但查询拆分失败，回退为原问题检索。",
        )

    def _is_complex_question(self, question: str) -> tuple[bool, str]:
        signals = []
        lowered = question.lower()

        if len(question.strip()) >= COMPLEX_LENGTH_THRESHOLD:
            signals.append("问题长度较长")

        connector_hits = sum(1 for token in COMPLEX_CONNECTORS if token in question)
        if connector_hits >= 1:
            signals.append("包含多个并列连接词")

        intent_hits = sum(1 for token in INTENT_MARKERS if token in question)
        if intent_hits >= 2:
            signals.append("同时包含多个提问意图")

        clause_count = len([part for part in re.split(r"[，,；;。?？]", question) if part.strip()])
        if clause_count >= 3:
            signals.append("包含多个语义分句")

        matched_topics = 0
        for keywords in TOPIC_KEYWORDS.values():
            if any(keyword.lower() in lowered for keyword in keywords):
                matched_topics += 1
        if matched_topics >= 2:
            signals.append("同时涉及多个睡眠主题")

        is_complex = len(signals) >= COMPLEXITY_SIGNAL_THRESHOLD
        if is_complex:
            return True, "；".join(signals)
        if signals:
            return False, f"仅检测到弱复杂信号：{'；'.join(signals)}"
        return False, "问题表达较集中，单次检索通常足够。"

    def _rewrite_with_llm(self, question: str) -> Optional[dict]:
        try:
            prompt = QUERY_REWRITE_PROMPT.format(
                question=question,
                max_sub_queries=MAX_SUB_QUERIES,
            )
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            parsed = json.loads(content)
            reason = (parsed.get("reason") or "LLM未提供原因").strip()
            sub_queries = parsed.get("sub_queries") or []
            cleaned_queries = self._clean_sub_queries(sub_queries, question)

            if not cleaned_queries:
                return None

            return {
                "reason": reason,
                "sub_queries": cleaned_queries,
            }
        except Exception:
            return None

    def _clean_sub_queries(self, sub_queries: List[str], original_question: str) -> List[str]:
        cleaned = []
        seen = set()

        for item in sub_queries[:MAX_SUB_QUERIES]:
            if not isinstance(item, str):
                continue
            query = re.sub(r"^\s*\d+[\.、]\s*", "", item).strip()
            if not query:
                continue
            if query == original_question:
                continue
            if query in seen:
                continue
            seen.add(query)
            cleaned.append(query)

        return cleaned
