import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from config import TOPIC_RULE_CONFIDENT_THRESHOLD

TOPIC_DEFINITIONS: Dict[str, str] = {
    "insomnia": "与入睡困难、早醒、易醒、睡眠维持困难、长期睡不着相关的问题。",
    "osa": "与打鼾、憋气、睡眠呼吸暂停、阻塞性睡眠呼吸暂停、白天嗜睡相关的问题。",
    "csa": "与中枢性睡眠呼吸暂停、潮式呼吸或中枢性通气异常相关的问题。",
    "sleep_hygiene": "与熬夜、作息不规律、睡前行为、睡眠习惯、改善睡眠方式相关的问题。",
    "general": "无法明确归入以上主题，或属于泛睡眠科普、混合问题、资料范围不清的问题。",
}

TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "insomnia": ["失眠", "入睡困难", "难以入睡", "很久才能睡着", "睡不着", "早醒", "易醒", "入眠困难", "醒得太早"],
    "osa": ["打鼾", "鼾", "呼吸暂停", "憋气", "阻塞性", "osa", "睡眠呼吸暂停", "白天犯困", "白天嗜睡"],
    "csa": ["中枢性", "csa", "潮式呼吸", "中枢性睡眠呼吸暂停"],
    "sleep_hygiene": ["睡眠卫生", "作息", "熬夜", "睡眠习惯", "改善睡眠", "睡眠不足", "倒时差", "睡前"],
}

CSA_PRIORITY_KEYWORDS = ["中枢性", "csa", "潮式呼吸", "中枢性睡眠呼吸暂停"]
RULE_CONFIDENT_THRESHOLD = TOPIC_RULE_CONFIDENT_THRESHOLD
TOPIC_OPTIONS = list(TOPIC_DEFINITIONS.keys())

TOPIC_ROUTER_PROMPT = PromptTemplate(
    template="""
你是睡眠健康问答系统的主题路由器，需要把用户问题归类到且仅归类到一个主题。

可选主题：
- insomnia: {insomnia}
- osa: {osa}
- csa: {csa}
- sleep_hygiene: {sleep_hygiene}
- general: {general}

分类规则：
1. 只能输出一个主题标签。
2. 如果问题主要在描述“入睡困难、早醒、易醒、睡不着”，优先归为 insomnia。
3. 如果问题主要在描述“打鼾、憋气、睡眠呼吸暂停、白天嗜睡”，优先归为 osa。
4. 如果问题明确提到“中枢性睡眠呼吸暂停、潮式呼吸”，归为 csa。
5. 如果问题主要询问“熬夜、作息、睡眠习惯、睡前行为、改善睡眠方式”，归为 sleep_hygiene。
6. 如果是混合问题、语义不清或无法确定，归为 general。
7. 输出必须是严格 JSON，格式如下：
{{"topic": "insomnia", "reason": "一句简短理由"}}

用户问题：{question}
""",
    input_variables=["question", *TOPIC_OPTIONS],
)


@dataclass
class TopicRouteResult:
    topic: str
    method: str
    reason: str
    rule_scores: Dict[str, int]


class TopicRouter:
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm

    def route(self, question: str) -> TopicRouteResult:
        rule_scores = self._score_by_rules(question)
        lowered = question.lower()

        if any(keyword.lower() in lowered for keyword in CSA_PRIORITY_KEYWORDS):
            return TopicRouteResult(
                topic="csa",
                method="rule",
                reason="命中 CSA 高优先级关键词",
                rule_scores=rule_scores,
            )

        rule_topic, rule_score = self._pick_best_rule_topic(rule_scores)

        if rule_score >= RULE_CONFIDENT_THRESHOLD:
            return TopicRouteResult(
                topic=rule_topic,
                method="rule",
                reason=f"规则命中 {rule_score} 个关键词",
                rule_scores=rule_scores,
            )

        if self.llm is not None:
            llm_result = self._route_with_llm(question, rule_scores)
            if llm_result is not None:
                return llm_result

        fallback_topic = rule_topic if rule_score > 0 else "general"
        fallback_reason = "规则弱命中，未启用或未获得LLM有效结果"
        return TopicRouteResult(
            topic=fallback_topic,
            method="rule_fallback",
            reason=fallback_reason,
            rule_scores=rule_scores,
        )

    def _score_by_rules(self, question: str) -> Dict[str, int]:
        lowered = question.lower()
        scores = {topic: 0 for topic in TOPIC_KEYWORDS}

        for topic, keywords in TOPIC_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in lowered:
                    scores[topic] += 1

        return scores

    def _pick_best_rule_topic(self, scores: Dict[str, int]) -> tuple[str, int]:
        if not scores:
            return "general", 0

        topic, score = max(scores.items(), key=lambda item: item[1])
        if score == 0:
            return "general", 0
        return topic, score

    def _route_with_llm(self, question: str, rule_scores: Dict[str, int]) -> Optional[TopicRouteResult]:
        try:
            prompt = TOPIC_ROUTER_PROMPT.format(question=question, **TOPIC_DEFINITIONS)
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            parsed = json.loads(content)
            topic = parsed.get("topic", "general")
            reason = parsed.get("reason", "LLM未提供原因")

            if topic not in TOPIC_OPTIONS:
                topic = "general"
                reason = f"LLM输出了无效主题，已回退为 general。原始原因：{reason}"

            return TopicRouteResult(
                topic=topic,
                method="llm",
                reason=reason,
                rule_scores=rule_scores,
            )
        except Exception as exc:
            return TopicRouteResult(
                topic="general",
                method="llm_fallback",
                reason=f"LLM分类失败，回退为 general：{exc}",
                rule_scores=rule_scores,
            )
