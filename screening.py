import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from topic_router import TOPIC_KEYWORDS

FOLLOW_UP_PROMPT = PromptTemplate(
    template="""
你是睡眠健康助手中的初筛追问模块。

请根据用户原始问题，生成用于“单轮补充信息”的追问问题。

要求：
1. 只输出严格 JSON。
2. follow_up_questions 必须是 2 到 4 个问题。
3. 问题要简洁、自然、容易回答，适合普通用户。
4. 优先追问对判断最重要的信息，例如：持续时间、发生频率、白天影响、是否有高风险表现、是否伴随其他关键症状。
5. 不要回答原问题，不要给建议，不要做诊断。
6. 不要重复提问，不要使用过于专业或吓人的措辞。
7. screening_type、risk_level、should_seek_care、reason 需要与用户问题保持一致。

输出格式：
{{
  "screening_type": "insomnia_like",
  "risk_level": "medium",
  "should_seek_care": false,
  "reason": "一句简短说明",
  "follow_up_questions": ["问题1", "问题2", "问题3"]
}}

可选 screening_type：
- insomnia_like
- breathing_related
- sleep_hygiene_like
- mixed
- unclear

可选 risk_level：
- low
- medium
- high

用户问题：{question}
规则初筛类型：{screening_type}
规则风险等级：{risk_level}
规则是否建议就医：{should_seek_care}
规则触发原因：{reason}
""",
    input_variables=["question", "screening_type", "risk_level", "should_seek_care", "reason"],
)

FOLLOW_UP_DECISION_PROMPT = PromptTemplate(
    template="""
你是睡眠健康助手中的初筛决策模块。

任务：
判断当前用户问题是否需要在正式回答前先做一轮补充追问。

要求：
1. 只输出严格 JSON。
2. 如果问题本身是知识型、定义型、标准型、比较型、指南/科普解读型，通常不需要追问。
3. 如果问题主要是症状主诉，且缺少持续时间、频率、白天影响或高风险表现等关键维度，通常需要追问。
4. 如果当前信息已经足够直接给出稳健回答，则不要追问。
5. 不要生成追问问题，只判断是否需要追问。

输出格式：
{{
  "needs_follow_up": false,
  "reason": "一句简短原因"
}}

用户问题：{question}
规则初筛类型：{screening_type}
规则风险等级：{risk_level}
规则是否建议就医：{should_seek_care}
规则触发原因：{reason}
""",
    input_variables=["question", "screening_type", "risk_level", "should_seek_care", "reason"],
)

SYMPTOM_MARKERS = [
    "睡不着",
    "入睡困难",
    "难以入睡",
    "更难入睡",
    "很久才能睡着",
    "睡着很慢",
    "躺下很久睡不着",
    "要一两个小时才能睡着",
    "早醒",
    "易醒",
    "醒得早",
    "半夜醒",
    "半夜两三点醒",
    "凌晨醒",
    "打鼾",
    "打呼噜",
    "呼噜很大",
    "憋气",
    "憋醒",
    "呼吸暂停",
    "喘不过气",
    "白天犯困",
    "白天嗜睡",
    "白天特别想睡",
    "困倦",
    "疲劳",
    "乏力",
    "没精神",
    "上班很累",
    "注意力不集中",
]
DURATION_MARKERS = ["多久", "持续", "一个月", "两周", "几周", "几个月", "半年", "一年", "长期", "最近", "三个月"]
FREQUENCY_MARKERS = ["每周", "每晚", "经常", "总是", "频繁", "偶尔", "几次", "每天", "一直", "老是", "很多"]
DAYTIME_IMPACT_MARKERS = ["白天", "疲劳", "困倦", "犯困", "注意力", "情绪", "工作", "学习", "精神状态", "没精神", "上班很累", "注意力差"]
RISK_MARKERS = ["憋醒", "呼吸暂停", "开车", "驾驶", "胸闷", "气短", "心慌", "情绪低落", "焦虑", "抑郁"]
KNOWLEDGE_MARKERS = ["指南", "怎么说", "是什么", "定义", "标准", "AASM", "中国睡眠研究会", "文档", "资料", "科普", "PSG", "OSA", "CSA"]
KNOWLEDGE_INTENT_MARKERS = ["是什么", "定义", "标准", "金标准", "诊断标准", "怎么说", "区别", "如何区分", "常见于哪些人", "哪些人", "危害", "建议", "推荐", "一种表现"]
DIAGNOSTIC_TEST_MARKERS = ["金标准", "确诊", "检查", "诊断标准", "怎么区分", "区别", "一线治疗", "首选治疗", "持续多久才算", "常见于哪些人"]
SOURCE_CONSTRAINT_MARKERS = ["AASM", "中国睡眠研究会", "中国医师协会", "中华医学会", "国家卫健委", "国家卫生健康委员会"]
BEHAVIOR_GUIDANCE_MARKERS = ["熬夜后", "睡前玩手机", "周末补觉", "晚上喝咖啡", "睡前运动", "怎么调整", "怎么改善", "好还是不好", "会不会影响", "能不能补回来"]
INDIVIDUAL_JUDGEMENT_MARKERS = ["是不是", "属于什么问题", "会不会有问题", "应该先看哪方面", "这算不算", "该怎么办", "怎么办", "会不会是", "什么情况", "一起存在"]
GENERIC_KNOWLEDGE_PATTERNS = ["一种表现", "会不会让人", "会让人", "常见于哪些人", "哪些人更容易", "需要做什么检查"]
PERSONAL_CONTEXT_MARKERS = ["我", "我的", "我们", "自己", "老公", "孩子", "家人", "最近", "已经"]
SCREENING_TOPIC_KEYWORDS = {
    "insomnia_like": ["睡着很慢", "躺下很久睡不着", "要一两个小时才能睡着", "半夜两三点醒", "凌晨醒"],
    "breathing_related": ["打呼噜", "呼噜很大", "喘不过气", "憋醒", "白天特别想睡"],
    "sleep_hygiene_like": ["熬夜", "补觉", "咖啡", "玩手机", "睡前运动"],
}
FOLLOW_UP_TYPE_OPTIONS = {"insomnia_like", "breathing_related", "sleep_hygiene_like", "mixed", "unclear"}
RISK_LEVEL_OPTIONS = {"low", "medium", "high"}


@dataclass
class ScreeningResult:
    needs_follow_up: bool
    screening_type: str
    risk_level: str
    should_seek_care: bool
    reason: str
    follow_up_questions: List[str]
    method: str


class ScreeningEngine:
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm

    def screen(self, question: str) -> ScreeningResult:
        screening_type = self._detect_screening_type(question)
        risk_level, should_seek_care = self._detect_risk(question)
        decision, reason = self._should_follow_up(question, screening_type)

        if decision == "uncertain" and self.llm is not None:
            llm_decision = self._decide_follow_up_with_llm(
                question=question,
                screening_type=screening_type,
                risk_level=risk_level,
                should_seek_care=should_seek_care,
                reason=reason,
            )
            if llm_decision is not None:
                needs_follow_up, llm_reason = llm_decision
                decision = "follow" if needs_follow_up else "no_follow"
                reason = f"规则：{reason}；LLM：{llm_reason}"
                method = "rule+llm"
            else:
                decision = "follow"
                method = "rule_fallback"
        else:
            method = "rule"

        needs_follow_up = decision == "follow"

        if not needs_follow_up:
            return ScreeningResult(
                needs_follow_up=False,
                screening_type=screening_type,
                risk_level=risk_level,
                should_seek_care=should_seek_care,
                reason=reason,
                follow_up_questions=[],
                method=method,
            )

        follow_up_questions = self.generate_follow_up_questions(
            question=question,
            screening_type=screening_type,
            risk_level=risk_level,
            should_seek_care=should_seek_care,
            reason=reason,
        )
        return ScreeningResult(
            needs_follow_up=True,
            screening_type=screening_type,
            risk_level=risk_level,
            should_seek_care=should_seek_care,
            reason=reason,
            follow_up_questions=follow_up_questions,
            method=method if self.llm is not None else "rule_fallback",
        )

    def generate_follow_up_questions(self, question: str, screening_type: str, risk_level: str, should_seek_care: bool, reason: str) -> List[str]:
        return self._generate_follow_up_questions(question, screening_type, risk_level, should_seek_care, reason)

    def _detect_screening_type(self, question: str) -> str:
        lowered = question.lower()
        topic_hits = {
            "insomnia_like": self._count_hits(lowered, TOPIC_KEYWORDS.get("insomnia", [])) + self._count_hits(lowered, SCREENING_TOPIC_KEYWORDS.get("insomnia_like", [])),
            "breathing_related": self._count_hits(lowered, TOPIC_KEYWORDS.get("osa", [])) + self._count_hits(lowered, TOPIC_KEYWORDS.get("csa", [])) + self._count_hits(lowered, SCREENING_TOPIC_KEYWORDS.get("breathing_related", [])),
            "sleep_hygiene_like": self._count_hits(lowered, TOPIC_KEYWORDS.get("sleep_hygiene", [])) + self._count_hits(lowered, SCREENING_TOPIC_KEYWORDS.get("sleep_hygiene_like", [])),
        }
        positive_topics = [name for name, score in topic_hits.items() if score > 0]
        if len(positive_topics) >= 2:
            return "mixed"
        if positive_topics:
            return max(topic_hits.items(), key=lambda item: item[1])[0]
        return "unclear"

    def _detect_question_intent(self, question: str, screening_type: str, symptom_hits: int, knowledge_hits: int, knowledge_intent_hits: int) -> str:
        lowered = question.lower()
        diagnostic_hits = self._count_hits(lowered, DIAGNOSTIC_TEST_MARKERS)
        source_hits = self._count_hits(lowered, SOURCE_CONSTRAINT_MARKERS)
        behavior_hits = self._count_hits(lowered, BEHAVIOR_GUIDANCE_MARKERS)
        individual_hits = self._count_hits(lowered, INDIVIDUAL_JUDGEMENT_MARKERS)
        generic_knowledge_hits = self._count_hits(lowered, GENERIC_KNOWLEDGE_PATTERNS)
        has_personal_context = self._count_hits(lowered, PERSONAL_CONTEXT_MARKERS) > 0

        if diagnostic_hits >= 1:
            return "diagnostic_test_or_standard"

        if source_hits >= 1 and (knowledge_hits >= 1 or knowledge_intent_hits >= 1):
            return "knowledge_explanation"

        if generic_knowledge_hits >= 1 and not has_personal_context:
            return "knowledge_explanation"

        if behavior_hits >= 1 and individual_hits == 0 and screening_type in {"sleep_hygiene_like", "unclear"}:
            return "knowledge_explanation"

        if knowledge_intent_hits >= 1 and symptom_hits <= 1 and individual_hits == 0:
            return "knowledge_explanation"

        if screening_type == "mixed":
            return "mixed_case"

        if symptom_hits >= 1 or individual_hits >= 1:
            return "symptom_complaint"

        if knowledge_hits >= 1 or knowledge_intent_hits >= 1:
            return "knowledge_explanation"

        return "unclear"

    def _detect_risk(self, question: str) -> tuple[str, bool]:
        lowered = question.lower()
        risk_hits = self._count_hits(lowered, RISK_MARKERS)
        if risk_hits >= 2:
            return "high", True
        if risk_hits == 1:
            return "medium", True
        if any(token in lowered for token in ["白天犯困", "白天嗜睡", "疲劳", "困倦"]):
            return "medium", False
        return "low", False

    def _should_follow_up(self, question: str, screening_type: str) -> tuple[str, str]:
        lowered = question.lower()
        symptom_hits = self._count_hits(lowered, SYMPTOM_MARKERS)
        knowledge_hits = self._count_hits(lowered, KNOWLEDGE_MARKERS)
        knowledge_intent_hits = self._count_hits(lowered, KNOWLEDGE_INTENT_MARKERS)
        individual_hits = self._count_hits(lowered, INDIVIDUAL_JUDGEMENT_MARKERS)
        behavior_hits = self._count_hits(lowered, BEHAVIOR_GUIDANCE_MARKERS)
        question_intent = self._detect_question_intent(
            question=question,
            screening_type=screening_type,
            symptom_hits=symptom_hits,
            knowledge_hits=knowledge_hits,
            knowledge_intent_hits=knowledge_intent_hits,
        )

        if question_intent == "diagnostic_test_or_standard":
            return "no_follow", "识别为检查/标准/鉴别类知识问题，可直接进入回答。"

        if question_intent == "knowledge_explanation":
            return "no_follow", "识别为知识检索或生活方式建议型问题，可直接进入回答。"

        if symptom_hits == 0 and screening_type == "unclear" and individual_hits == 0:
            return "no_follow", "未识别到明确症状型主诉，可直接进入回答。"

        missing_dimensions = []
        if self._count_hits(lowered, DURATION_MARKERS) == 0:
            missing_dimensions.append("持续时间")
        if self._count_hits(lowered, FREQUENCY_MARKERS) == 0:
            missing_dimensions.append("发生频率")
        if self._count_hits(lowered, DAYTIME_IMPACT_MARKERS) == 0:
            missing_dimensions.append("白天影响")

        if screening_type in {"insomnia_like", "breathing_related", "mixed"} and symptom_hits >= 1 and len(missing_dimensions) >= 2:
            return "follow", f"识别到睡眠症状主诉，但缺少{ '、'.join(missing_dimensions[:3]) }等关键信息。"

        if screening_type in {"insomnia_like", "breathing_related", "mixed"} and individual_hits >= 1 and len(missing_dimensions) >= 1:
            return "follow", f"用户在请求个体化判断，但仍缺少{ '、'.join(missing_dimensions[:3]) }等关键信息。"

        if screening_type == "mixed" and individual_hits >= 1:
            return "follow", "识别为多方向混合问题，建议先补充关键背景后再回答。"

        if screening_type == "sleep_hygiene_like" and symptom_hits == 0 and (knowledge_hits >= 1 or knowledge_intent_hits >= 1):
            return "no_follow", "识别为睡眠卫生知识/建议型问题，可直接进入回答。"

        if screening_type == "sleep_hygiene_like" and symptom_hits >= 1 and individual_hits >= 1:
            return "follow", "识别为作息问题伴随个体化症状判断，建议先补充关键背景。"

        if screening_type == "sleep_hygiene_like" and behavior_hits >= 1:
            return "no_follow", "识别为行为习惯影响与调整建议问题，可直接进入回答。"

        if screening_type == "sleep_hygiene_like" and len(question.strip()) <= 24:
            return "uncertain", "识别到作息/睡眠习惯相关问题，但需要进一步判断是否应先补充信息。"

        if symptom_hits >= 1 and len(question.strip()) <= 20:
            return "uncertain", "问题较短且包含症状描述，需进一步判断是否先补充信息。"

        if knowledge_hits >= 1 or knowledge_intent_hits >= 1:
            return "uncertain", "问题包含知识型表达，但同时存在睡眠相关信号，需进一步判断是否追问。"

        return "no_follow", "当前问题信息相对充分，可直接进入回答。"

    def _generate_follow_up_questions(self, question: str, screening_type: str, risk_level: str, should_seek_care: bool, reason: str) -> List[str]:
        if self.llm is not None:
            try:
                prompt = FOLLOW_UP_PROMPT.format(
                    question=question,
                    screening_type=screening_type,
                    risk_level=risk_level,
                    should_seek_care=str(should_seek_care).lower(),
                    reason=reason,
                )
                response = self.llm.invoke(prompt)
                parsed = json.loads(response.content.strip())
                llm_questions = self._clean_questions(parsed.get("follow_up_questions") or [])
                if llm_questions:
                    return llm_questions
            except Exception:
                pass
        return self._fallback_questions(screening_type)

    def _decide_follow_up_with_llm(self, question: str, screening_type: str, risk_level: str, should_seek_care: bool, reason: str) -> Optional[tuple[bool, str]]:
        try:
            prompt = FOLLOW_UP_DECISION_PROMPT.format(
                question=question,
                screening_type=screening_type,
                risk_level=risk_level,
                should_seek_care=str(should_seek_care).lower(),
                reason=reason,
            )
            response = self.llm.invoke(prompt)
            parsed = json.loads(response.content.strip())
            needs_follow_up = bool(parsed.get("needs_follow_up"))
            llm_reason = (parsed.get("reason") or "LLM未提供原因").strip()
            return needs_follow_up, llm_reason
        except Exception:
            return None

    def _fallback_questions(self, screening_type: str) -> List[str]:
        fallback_map = {
            "insomnia_like": [
                "这种情况大概持续多久了？",
                "一周大约会出现几次？",
                "白天会不会疲劳、注意力下降或情绪受影响？",
            ],
            "breathing_related": [
                "这种情况大概持续多久了？",
                "睡觉时是否会憋醒，或者被别人发现有呼吸暂停？",
                "白天会不会明显犯困，甚至影响开车、工作或学习？",
            ],
            "sleep_hygiene_like": [
                "你最近的作息大概是怎样的？",
                "睡前是否经常玩手机、喝咖啡或熬夜？",
                "白天状态是否已经受到影响？",
            ],
            "mixed": [
                "这些情况大概持续多久了？",
                "一周大约会出现几次，最困扰你的表现是什么？",
                "白天是否已经受到明显影响，例如疲劳、犯困或注意力下降？",
            ],
            "unclear": [
                "你最想解决的睡眠问题具体是什么？",
                "这种情况大概持续多久了？",
                "白天状态是否已经受到影响？",
            ],
        }
        return fallback_map.get(screening_type, fallback_map["unclear"])

    def _clean_questions(self, questions: List[str]) -> List[str]:
        cleaned: List[str] = []
        seen = set()
        for item in questions[:4]:
            if not isinstance(item, str):
                continue
            question = item.strip()
            if not question or question in seen:
                continue
            seen.add(question)
            cleaned.append(question)
        return cleaned[:4]

    def _count_hits(self, text: str, keywords: List[str]) -> int:
        return sum(1 for keyword in keywords if keyword.lower() in text)
