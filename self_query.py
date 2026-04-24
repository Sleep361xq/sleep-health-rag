import json
from dataclasses import dataclass
from typing import Dict, Optional

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from topic_router import TOPIC_KEYWORDS

DOC_TYPE_RULES = {
    "guideline": ["指南", "共识", "诊疗规范", "临床指南"],
    "education": ["科普", "科普资料", "健康宣教", "睡眠科普"],
    "qa": ["问答", "问答库", "qa", "常见问题"],
}

IMPLICIT_GUIDELINE_MARKERS = [
    "金标准",
    "确诊",
    "检查",
    "诊断",
    "诊断标准",
    "分级",
    "一线治疗",
    "首选治疗",
    "怎么区分",
    "区别",
    "持续多久才算",
    "常见于哪些人",
]

IMPLICIT_EDUCATION_MARKERS = [
    "影响",
    "危害",
    "后果",
    "怎么调整",
    "怎么改善",
    "好还是不好",
    "会不会",
    "能不能",
    "补回来",
]

GUIDELINE_FRIENDLY_SOURCES = {
    "AASM",
    "中华医学会神经病学分会睡眠障碍学组",
    "中国医师协会睡眠医学专业委员会",
    "医药专论",
}

EDUCATION_FRIENDLY_SOURCES = {
    "中国睡眠研究会",
    "中国国家卫生健康委员会",
}

SOURCE_RULES = {
    "AASM": ["aasm", "美国睡眠医学会"],
    "中国睡眠研究会": ["中国睡眠研究会"],
    "中华医学会神经病学分会睡眠障碍学组": ["中华医学会", "神经病学分会睡眠障碍学组"],
    "中国医师协会睡眠医学专业委员会": ["中国医师协会睡眠医学专业委员会", "中国医师协会"],
    "中国国家卫生健康委员会": ["国家卫健委", "国家卫生健康委员会", "中国国家卫生健康委员会"],
    "医药专论": ["医药专论"],
}

SELF_QUERY_TRIGGER_WORDS = [
    "指南",
    "科普",
    "问答",
    "问答库",
    "来源",
    "机构",
    "学会",
    "哪篇",
    "哪一篇",
    "官方",
    "权威",
    "aasm",
    "美国睡眠医学会",
    "中国睡眠研究会",
    "中华医学会",
    "标准",
    "金标准",
    "诊断标准",
    "区别",
    "如何区分",
    "常见于哪些人",
    "哪些人",
    "危害",
    "建议",
    "推荐",
    "影响",
    "后果",
    "怎么调整",
    "怎么改善",
    "好还是不好",
    "会不会",
    "能不能",
]

KNOWLEDGE_STYLE_MARKERS = [
    "是什么",
    "定义",
    "标准",
    "金标准",
    "诊断标准",
    "区别",
    "如何区分",
    "常见于哪些人",
    "哪些人",
    "危害",
    "建议",
    "推荐",
    "影响",
    "后果",
    "怎么调整",
    "怎么改善",
    "好还是不好",
    "会不会",
    "能不能",
]

SELF_QUERY_PROMPT = PromptTemplate(
    template="""
你是睡眠健康RAG系统的 self-query 解析器。

任务：
从用户问题中提取可用于 metadata 过滤的检索约束。只允许提取以下字段：
- doc_type: 可选值 guideline / education / qa
- source: 来源机构名称
- topic: 可选值 insomnia / osa / csa / sleep_hygiene / general

要求：
1. 只有当用户问题中存在明确或高度隐含的约束时才填写。
2. 如果无法确定，字段必须为 null。
3. 不要臆测，不要为了凑字段而填写。
4. 输出必须是严格 JSON，格式如下：
{{
  "doc_type": "guideline",
  "source": "AASM",
  "topic": "osa",
  "reason": "一句简短原因"
}}

用户问题：{question}
""",
    input_variables=["question"],
)


@dataclass
class SelfQueryResult:
    doc_type: Optional[str]
    source: Optional[str]
    topic: Optional[str]
    method: str
    reason: str


class SelfQueryParser:
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm

    def parse(self, question: str) -> SelfQueryResult:
        rule_result = self._parse_by_rules(question)
        if self._is_sufficient(rule_result):
            return rule_result

        if self._should_use_llm(question, rule_result) and self.llm is not None:
            llm_result = self._parse_with_llm(question)
            if llm_result is not None:
                merged = SelfQueryResult(
                    doc_type=rule_result.doc_type or llm_result.doc_type,
                    source=rule_result.source or llm_result.source,
                    topic=rule_result.topic or llm_result.topic,
                    method="rule+llm" if rule_result.method == "rule" else llm_result.method,
                    reason=f"规则：{rule_result.reason}；LLM：{llm_result.reason}",
                )
                return merged

        return rule_result

    def _parse_by_rules(self, question: str) -> SelfQueryResult:
        lowered = question.lower()
        doc_type = None
        source = None
        topic = None
        reasons = []

        for mapped_doc_type, keywords in DOC_TYPE_RULES.items():
            if any(keyword.lower() in lowered for keyword in keywords):
                doc_type = mapped_doc_type
                reasons.append(f"识别到文档类型 {mapped_doc_type}")
                break

        for mapped_source, keywords in SOURCE_RULES.items():
            if any(keyword.lower() in lowered for keyword in keywords):
                source = mapped_source
                reasons.append(f"识别到来源 {mapped_source}")
                break

        topic_hits: Dict[str, int] = {}
        for mapped_topic, keywords in TOPIC_KEYWORDS.items():
            hit_count = sum(1 for keyword in keywords if keyword.lower() in lowered)
            if hit_count > 0:
                topic_hits[mapped_topic] = hit_count

        if topic_hits:
            topic = max(topic_hits.items(), key=lambda item: item[1])[0]
            reasons.append(f"识别到主题 {topic}")

        if doc_type is None:
            inferred_doc_type = self._infer_doc_type(lowered, source, topic)
            if inferred_doc_type is not None:
                doc_type = inferred_doc_type
                reasons.append(f"识别到隐式文档类型 {inferred_doc_type}")

        if doc_type or source or topic:
            return SelfQueryResult(
                doc_type=doc_type,
                source=source,
                topic=topic,
                method="rule",
                reason="；".join(reasons),
            )

        return SelfQueryResult(
            doc_type=None,
            source=None,
            topic=None,
            method="none",
            reason="未检测到明确 metadata 检索约束。",
        )

    def _is_sufficient(self, result: SelfQueryResult) -> bool:
        extracted_count = sum(1 for value in [result.doc_type, result.source, result.topic] if value)
        return extracted_count >= 2

    def _should_use_llm(self, question: str, result: SelfQueryResult) -> bool:
        lowered = question.lower()
        has_trigger = any(token.lower() in lowered for token in SELF_QUERY_TRIGGER_WORDS)
        has_knowledge_style = any(token.lower() in lowered for token in KNOWLEDGE_STYLE_MARKERS)
        has_implicit_guideline = any(token.lower() in lowered for token in IMPLICIT_GUIDELINE_MARKERS)
        has_implicit_education = any(token.lower() in lowered for token in IMPLICIT_EDUCATION_MARKERS)
        extracted_count = sum(1 for value in [result.doc_type, result.source, result.topic] if value)

        if has_trigger and extracted_count <= 1:
            return True

        if has_knowledge_style and extracted_count <= 1:
            return True

        if result.topic is not None and result.doc_type is None and (has_knowledge_style or has_implicit_guideline or has_implicit_education):
            return True

        return False

    def _infer_doc_type(self, lowered: str, source: Optional[str], topic: Optional[str]) -> Optional[str]:
        guideline_hits = sum(1 for token in IMPLICIT_GUIDELINE_MARKERS if token.lower() in lowered)
        education_hits = sum(1 for token in IMPLICIT_EDUCATION_MARKERS if token.lower() in lowered)

        if source in GUIDELINE_FRIENDLY_SOURCES and ("建议" in lowered or "怎么说" in lowered):
            return "guideline"

        if source in EDUCATION_FRIENDLY_SOURCES and ("科普" in lowered or education_hits >= 1):
            return "education"

        if guideline_hits > education_hits and guideline_hits >= 1:
            return "guideline"

        if education_hits > guideline_hits and education_hits >= 1:
            return "education"

        if guideline_hits >= 1 and topic in {"insomnia", "osa", "csa"}:
            return "guideline"

        if education_hits >= 1:
            return "education"

        return None

    def _parse_with_llm(self, question: str) -> Optional[SelfQueryResult]:
        try:
            prompt = SELF_QUERY_PROMPT.format(question=question)
            response = self.llm.invoke(prompt)
            parsed = json.loads(response.content.strip())
            doc_type = parsed.get("doc_type")
            source = parsed.get("source")
            topic = parsed.get("topic")
            reason = parsed.get("reason", "LLM未提供原因")

            if doc_type not in {"guideline", "education", "qa", None}:
                doc_type = None
            if topic not in {"insomnia", "osa", "csa", "sleep_hygiene", "general", None}:
                topic = None
            if isinstance(source, str):
                source = source.strip() or None
            else:
                source = None

            return SelfQueryResult(
                doc_type=doc_type,
                source=source,
                topic=topic,
                method="llm",
                reason=reason,
            )
        except Exception:
            return None
