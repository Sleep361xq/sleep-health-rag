import html
import re
import tempfile
from pathlib import Path

import streamlit as st

from analysis_tools import analyze_uploaded_sleep_data, bundle_to_dict
from config import APP_PORT
from query_rewriter import QueryRewriter
from rag_router import answer_question, build_llm, build_reranker, build_vectorstore
from screening import ScreeningEngine
from self_query import SelfQueryParser
from topic_router import TopicRouter

st.set_page_config(page_title="睡眠健康智能问答系统", page_icon="💤", layout="wide", initial_sidebar_state="expanded")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=ZCOOL+XiaoWei&family=Noto+Sans+SC:wght@400;500;700&display=swap');
:root { --bg-main:#06131a; --bg-panel:rgba(9,28,38,.88); --bg-card:rgba(15,39,51,.92); --line-soft:rgba(129,201,149,.18); --line-strong:rgba(129,201,149,.42); --text-main:#e8f3ee; --text-muted:#a8c6b9; --text-sidebar:#d7ebe2; --accent:#86d7a2; --accent-2:#4eb6d5; --warning:#f0c674; }
html, body, [data-testid="stAppViewContainer"] { background: radial-gradient(circle at 20% 20%, rgba(78,182,213,.12), transparent 28%), radial-gradient(circle at 80% 15%, rgba(134,215,162,.10), transparent 24%), linear-gradient(180deg, #041017 0%, #071720 45%, #08141d 100%); color: var(--text-main); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(8,26,36,.96), rgba(5,16,24,.98)); border-right: 1px solid var(--line-soft); }
[data-testid="stSidebar"] * { color: var(--text-sidebar); }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
.hero-card { background: linear-gradient(135deg, rgba(13,36,49,.92), rgba(6,22,31,.95)); border:1px solid var(--line-strong); border-radius:24px; padding:1.6rem 1.8rem; box-shadow:0 20px 60px rgba(0,0,0,.28); margin-bottom:1.2rem; }
.hero-title { font-family:'ZCOOL XiaoWei', serif; font-size:2.2rem; line-height:1.2; color:var(--text-main); margin-bottom:.4rem; }
.hero-subtitle { font-family:'Noto Sans SC', sans-serif; color:var(--text-muted); font-size:1rem; }
.info-chip-wrap { display:flex; flex-wrap:wrap; gap:.6rem; margin-top:1rem; }
.info-chip { padding:.45rem .8rem; border-radius:999px; border:1px solid var(--line-soft); background:rgba(134,215,162,.08); color:var(--accent); font-size:.9rem; }
.section-title { font-family:'Noto Sans SC', 'Microsoft YaHei', 'PingFang SC', sans-serif; color:var(--text-main); font-size:1.2rem; margin-bottom:.8rem; }
[data-testid="stVerticalBlockBorderWrapper"] { background:var(--bg-panel); border:1px solid var(--line-soft); border-radius:20px; padding:.35rem; box-shadow:0 16px 40px rgba(0,0,0,.18); }
.metric-grid { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:.8rem; margin:1rem 0 .8rem 0; }
.metric-card { background:var(--bg-card); border:1px solid var(--line-soft); border-radius:18px; padding:.9rem 1rem; }
.metric-label { color:var(--text-muted); font-size:.82rem; margin-bottom:.35rem; }
.metric-value { color:var(--accent); font-size:1rem; font-weight:700; }
.answer-html-card { background:linear-gradient(180deg, rgba(14,38,49,.94), rgba(9,25,33,.96)); border:1px solid var(--line-strong); border-radius:22px; padding:1.1rem 1.2rem; margin-top:.4rem; }
.answer-meta { color:var(--text-muted); font-size:.92rem; margin-bottom:.65rem; }
.answer-content { color:var(--text-main); line-height:1.8; font-size:.98rem; }
.answer-paragraph { margin:0 0 .7rem 0; }
.answer-paragraph:last-child { margin-bottom:0; }
.answer-content ul, .answer-content ol { padding-left:1.3rem; }
.ref-card, .rewrite-card, .rerank-card, .self-query-card, .screening-card, .followup-card { background:rgba(8,24,31,.94); border:1px solid var(--line-soft); border-radius:18px; padding:.95rem 1rem; margin-bottom:.8rem; }
.ref-title, .rewrite-title, .rerank-title, .self-query-title, .screening-title, .followup-title { color:var(--accent-2); font-weight:700; margin-bottom:.35rem; line-height:1.5; }
.ref-meta, .rewrite-meta, .rerank-meta, .self-query-meta, .screening-meta, .followup-meta { color:var(--text-muted); font-size:.85rem; margin-bottom:.35rem; line-height:1.6; }
.followup-list { color:var(--text-main); font-size:.92rem; line-height:1.7; padding-left:1.1rem; margin:.25rem 0 0 0; }
.ref-section { color:var(--accent); font-size:.84rem; margin-bottom:.45rem; line-height:1.6; }
.ref-content { color:var(--text-main); font-size:.92rem; line-height:1.65; }
.rewrite-list { color:var(--text-main); font-size:.92rem; line-height:1.7; padding-left:1.1rem; margin:.2rem 0 0 0; }
.disclaimer { margin-top:1rem; border-left:4px solid var(--warning); padding:.8rem 1rem; background:rgba(240,198,116,.08); border-radius:12px; color:#f7ddb0; font-size:.92rem; }
[data-testid="stTextArea"] textarea { background:rgba(6,24,32,.95)!important; color:var(--text-main)!important; border-radius:16px!important; border:1px solid var(--line-soft)!important; font-size:1rem!important; }
.stButton > button { width:100%; border-radius:14px; border:1px solid rgba(134,215,162,.35); background:linear-gradient(90deg, #78cc98, #4eb6d5); color:#041017; font-weight:700; padding:.75rem 1rem; }
</style>
"""

EXAMPLE_QUESTIONS = [
    "我每天晚上要躺下一两个小时才能入睡，这是什么症状，怎么解决？",
    "AASM 关于 OSA 的指南怎么说？",
    "中国睡眠研究会的科普里，睡眠不足有什么危害？",
    "长期失眠同时白天疲倦、偶尔打鼾，可能是什么问题，应该怎么处理？",
]


def save_uploaded_npz(uploaded_file, prefix: str) -> str:
    suffix = Path(uploaded_file.name).suffix or ".npz"
    safe_stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(uploaded_file.name).stem) or prefix
    temp_path = Path(tempfile.gettempdir()) / f"{prefix}_{safe_stem}{suffix}"
    temp_path.write_bytes(uploaded_file.getvalue())
    return str(temp_path)


@st.cache_resource(show_spinner=False)
def load_runtime_resources():
    llm = build_llm()
    vectordb = build_vectorstore()
    topic_router = TopicRouter(llm=llm)
    query_rewriter = QueryRewriter(llm=llm)
    reranker = build_reranker()
    self_query_parser = SelfQueryParser(llm=llm)
    screening_engine = ScreeningEngine(llm=llm)
    return llm, vectordb, topic_router, query_rewriter, reranker, self_query_parser, screening_engine


def render_reference_card(doc, index: int):
    metadata = doc.metadata
    title = metadata.get("title", metadata.get("filename", "未知文档"))
    source = metadata.get("source", metadata.get("filename", "未知来源"))
    doc_type = metadata.get("doc_type", "unknown")
    qa_doc_type = metadata.get("qa_doc_type", "")
    topic = metadata.get("topic", "unknown")
    section_path = metadata.get("section_path", "")
    qa_source = metadata.get("qa_source", "")
    rerank_score = metadata.get("rerank_score", "")
    card_title = f"证据 {index} · {title}"
    meta_parts = [f"类型：{doc_type}"]
    if qa_doc_type:
        meta_parts.append(f"QA类型：{qa_doc_type}")
    meta_parts.extend([f"主题：{topic}", f"来源：{source}"])
    if qa_source:
        meta_parts.append(f"QA来源：{qa_source}")
    if rerank_score:
        meta_parts.append(f"重排序分数：{rerank_score}")
    preview = doc.page_content.strip().replace("\n", " ")
    if len(preview) > 220:
        preview = preview[:220] + " ..."
    section_html = f'<div class="ref-section">章节：{section_path}</div>' if section_path else ""
    st.markdown(f"""
<div class="ref-card">
  <div class="ref-title">{card_title}</div>
  <div class="ref-meta">{' ｜ '.join(meta_parts)}</div>
  {section_html}
  <div class="ref-content">{preview}</div>
</div>
""", unsafe_allow_html=True)


def render_query_rewrite_card(result: dict):
    rewrite = result.get("query_rewrite", {})
    if not rewrite:
        return
    sub_query_topics = result.get("sub_query_topics", [])
    bullet_items = []
    if rewrite.get("is_complex") and sub_query_topics:
        for item in sub_query_topics:
            bullet_items.append(f"<li>{item['query']}<br><span class='rewrite-meta'>主题：{item['topic']} ｜ 路由方式：{item['route_method']}</span></li>")
    else:
        for query in rewrite.get("rewritten_queries", []):
            bullet_items.append(f"<li>{query}</li>")
    st.markdown(f"""
<div class="rewrite-card">
  <div class="rewrite-title">查询重构 / 多查询检索</div>
  <div class="rewrite-meta"><strong>是否复杂问题：</strong> {rewrite.get('is_complex')}</div>
  <div class="rewrite-meta"><strong>复杂度判断：</strong> {rewrite.get('complexity_reason')}</div>
  <div class="rewrite-meta"><strong>改写方式：</strong> {rewrite.get('rewrite_method')} ｜ <strong>说明：</strong> {rewrite.get('rewrite_reason')}</div>
  <ul class="rewrite-list">{''.join(bullet_items)}</ul>
</div>
""", unsafe_allow_html=True)


def render_self_query_card(result: dict):
    info = result.get("self_query", {})
    if not info:
        return
    st.markdown(f"""
<div class="self-query-card">
  <div class="self-query-title">Self-query / 元数据约束</div>
  <div class="self-query-meta"><strong>doc_type：</strong> {info.get('doc_type')}</div>
  <div class="self-query-meta"><strong>source：</strong> {info.get('source')}</div>
  <div class="self-query-meta"><strong>topic：</strong> {info.get('topic')}</div>
  <div class="self-query-meta"><strong>解析方式：</strong> {info.get('method')}</div>
  <div class="self-query-meta"><strong>说明：</strong> {info.get('reason')}</div>
</div>
""", unsafe_allow_html=True)


def render_screening_card(screening: dict):
    if not screening:
        return
    st.markdown(f"""
<div class="screening-card">
  <div class="screening-title">初筛结果 / Screening</div>
  <div class="screening-meta"><strong>是否需要追问：</strong> {screening.get('needs_follow_up')}</div>
  <div class="screening-meta"><strong>主诉类型：</strong> {screening.get('screening_type')}</div>
  <div class="screening-meta"><strong>风险等级：</strong> {screening.get('risk_level')}</div>
  <div class="screening-meta"><strong>是否建议尽快就医：</strong> {screening.get('should_seek_care')}</div>
  <div class="screening-meta"><strong>判断方式：</strong> {screening.get('method')}</div>
  <div class="screening-meta"><strong>说明：</strong> {screening.get('reason')}</div>
</div>
""", unsafe_allow_html=True)


def render_follow_up_card(screening: dict):
    questions = screening.get("follow_up_questions", []) if screening else []
    if not questions:
        return
    items = "".join([f"<li>{html.escape(q)}</li>" for q in questions])
    st.markdown(f"""
<div class="followup-card">
  <div class="followup-title">补充信息追问 / Follow-up</div>
  <div class="followup-meta">请一次性补充下面这些关键信息，系统会结合你的补充再生成最终回答。</div>
  <ol class="followup-list">{items}</ol>
</div>
""", unsafe_allow_html=True)


def render_follow_up_decision_card(screening: dict):
    if not screening:
        return
    st.markdown(f"""
<div class="followup-card">
  <div class="followup-title">是否需要补充信息</div>
  <div class="followup-meta">系统判断当前问题可以直接回答。若你希望系统先进一步了解情况，也可以先进行一次追问，再生成更有针对性的回答。</div>
  <div class="followup-meta"><strong>当前说明：</strong> {screening.get('reason')}</div>
</div>
""", unsafe_allow_html=True)


def render_rerank_card(result: dict):
    rerank = result.get("rerank", {})
    if not rerank:
        return
    st.markdown(f"""
<div class="rerank-card">
  <div class="rerank-title">重排序 / Rerank</div>
  <div class="rerank-meta"><strong>是否启用：</strong> {rerank.get('enabled')}</div>
  <div class="rerank-meta"><strong>候选文档数：</strong> {rerank.get('candidate_count')}</div>
  <div class="rerank-meta"><strong>最终指南/科普：</strong> {rerank.get('final_pdf_count')} ｜ <strong>最终QA：</strong> {rerank.get('final_qa_count')}</div>
  <div class="rerank-meta"><strong>最少保留指南数：</strong> {rerank.get('guideline_min')}</div>
</div>
""", unsafe_allow_html=True)


def render_answer_card(result: dict):
    answer_text = result["answer"]
    answer_text = re.sub(r"`\s*\[", "[", answer_text)
    answer_text = re.sub(r"\]\s*`", "]", answer_text)
    answer_text = answer_text.replace("`", "")
    answer_text = re.sub(r"\n{3,}", "\n\n", answer_text).strip()

    paragraphs = []
    for block in answer_text.split("\n\n"):
        safe_block = html.escape(block).replace("\n", "<br>")
        paragraphs.append(f'<div class="answer-paragraph">{safe_block}</div>')

    answer_html = "".join(paragraphs)
    st.markdown(f"""
<div class="answer-html-card">
  <div class="section-title">系统回答</div>
  <div class="answer-meta"><strong>路由说明：</strong> {result['route_reason']}</div>
  <div class="answer-meta"><strong>规则得分：</strong> {result['rule_scores']}</div>
  <div class="answer-content">{answer_html}</div>
  <div class="disclaimer">本系统面向睡眠健康场景提供知识检索与问答支持，回答基于现有指南、科普资料与问答库生成，仅供健康参考，不替代临床诊断。</div>
</div>
""", unsafe_allow_html=True)


def render_analysis_card(analysis: dict):
    if not analysis:
        return
    st.markdown('<div class="section-title">睡眠数据分析结果</div>', unsafe_allow_html=True)

    staging = analysis.get("staging")
    if staging:
        counts = staging.get("stage_counts", {})
        ratios = staging.get("stage_ratios", {})
        counts_text = " ｜ ".join([f"{stage}: {count}" for stage, count in counts.items()]) if counts else "暂无"
        ratios_text = " ｜ ".join([f"{stage}: {ratio:.1%}" for stage, ratio in ratios.items()]) if ratios else "暂无"
        st.markdown(f"""
<div class="screening-card">
  <div class="screening-title">睡眠分期工具</div>
  <div class="screening-meta"><strong>总 epoch 数：</strong> {staging.get('epoch_count')}</div>
  <div class="screening-meta"><strong>分期计数：</strong> {counts_text}</div>
  <div class="screening-meta"><strong>分期占比：</strong> {ratios_text}</div>
</div>
""", unsafe_allow_html=True)

    osa = analysis.get("osa")
    if osa:
        probs = osa.get("class_probabilities", {})
        probs_text = " ｜ ".join([f"{label}: {prob:.3f}" for label, prob in probs.items()]) if probs else "暂无"
        st.markdown(f"""
<div class="screening-card">
  <div class="screening-title">OSA 分类工具</div>
  <div class="screening-meta"><strong>预测分级：</strong> {osa.get('severity_label')}</div>
  <div class="screening-meta"><strong>类别概率：</strong> {probs_text}</div>
</div>
""", unsafe_allow_html=True)

    summary = analysis.get("summary", "")
    if summary:
        st.caption("分析摘要会自动带入后续问答，帮助系统结合个体数据进行解释。")
        st.code(summary, language="text")


def build_augmented_question(question: str, follow_up_notes: str) -> str:
    notes = follow_up_notes.strip()
    if not notes:
        return question.strip()
    return f"原始问题：\n{question.strip()}\n\n补充信息：\n{notes}"


def build_analysis_augmented_question(question: str, analysis_summary: str, follow_up_notes: str = "") -> str:
    parts = [f"原始问题：\n{question.strip()}"]
    if analysis_summary.strip():
        parts.append(f"个体睡眠数据分析结果：\n{analysis_summary.strip()}")
    if follow_up_notes.strip():
        parts.append(f"补充信息：\n{follow_up_notes.strip()}")
    return "\n\n".join(parts)


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown("""
<div class="hero-card">
  <div class="hero-title">睡眠健康智能问答系统</div>
  <div class="hero-subtitle">基于指南、科普资料与结构化问答库构建的垂直领域 RAG 原型，支持主题路由、查询重构、Self-query、重排序与可溯源回答。</div>
  <div class="info-chip-wrap">
    <div class="info-chip">规则 + LLM 主题路由</div>
    <div class="info-chip">复杂问题多查询检索</div>
    <div class="info-chip">自定义 Self-query</div>
    <div class="info-chip">本地 HuggingFace Rerank</div>
    <div class="info-chip">指南优先检索</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.session_state.setdefault("latest_result", None)
    st.session_state.setdefault("screening_result", None)
    st.session_state.setdefault("pending_question", "")
    st.session_state.setdefault("question_input", EXAMPLE_QUESTIONS[0])
    st.session_state.setdefault("follow_up_mode", None)
    st.session_state.setdefault("latest_analysis", {})

    with st.sidebar:
        st.markdown("## 使用说明")
        st.markdown(f"- 当前默认服务端口：`{APP_PORT}`\n- 输入一个与你睡眠相关的问题\n- 系统会先做睡眠问题初筛，必要时生成单轮追问\n- 若系统判断可直接回答，你也可以主动选择先追问一次\n- 补充信息后，系统会再做查询重构、Self-query 与重排序\n- 回答仅供健康参考，不替代临床诊断")
        selected_example = st.selectbox("示例问题", ["请选择一个示例问题"] + EXAMPLE_QUESTIONS)
        if selected_example != "请选择一个示例问题":
            st.session_state["question_input"] = selected_example

        st.markdown("---")
        st.markdown("## 睡眠数据分析工具")
        st.caption("上传你的 `.npz` 文件后，可先运行睡眠分期和 OSA 分类，再结合当前问题做结果解读。")
        staging_file = st.file_uploader("上传 EEG/EOG 睡眠分期文件", type=["npz"], key="staging_npz")
        osa_file = st.file_uploader("上传 OSA 分类输入文件", type=["npz"], key="osa_npz")
        analyze_clicked = st.button("运行工具分析")
        if analyze_clicked and (staging_file or osa_file):
            with st.spinner("正在调用睡眠分期与 OSA 分类工具..."):
                try:
                    staging_path = save_uploaded_npz(staging_file, "staging") if staging_file else None
                    osa_path = save_uploaded_npz(osa_file, "osa") if osa_file else None
                    analysis_bundle = analyze_uploaded_sleep_data(staging_npz_path=staging_path, osa_npz_path=osa_path)
                    st.session_state["latest_analysis"] = bundle_to_dict(analysis_bundle)
                    completed_tools = []
                    if staging_file:
                        completed_tools.append("睡眠分期")
                    if osa_file:
                        completed_tools.append("OSA 分类")
                    st.success(f"工具分析完成：{'、'.join(completed_tools)}")
                except Exception as exc:
                    st.session_state["latest_analysis"] = {}
                    st.error(f"工具分析失败：{exc}")
        elif analyze_clicked:
            st.warning("请至少上传一个 `.npz` 文件后再运行分析。")

    default_question = st.session_state.get("question_input", EXAMPLE_QUESTIONS[0])
    left_col, right_col = st.columns([1.35, 1], gap="large")

    with left_col:
        with st.container(border=True):
            st.markdown('<div class="section-title">问题输入</div>', unsafe_allow_html=True)
            question = st.text_area("请输入你的问题", value=default_question, height=150, label_visibility="collapsed", key="question_box")
            ask = st.button("开始初筛")

        analysis = st.session_state.get("latest_analysis") or {}
        if analysis:
            with st.container(border=True):
                render_analysis_card(analysis)

        screening = st.session_state.get("screening_result")
        if ask and question.strip():
            with st.spinner("正在进行睡眠问题初筛与追问判断..."):
                llm, vectordb, topic_router, query_rewriter, reranker, self_query_parser, screening_engine = load_runtime_resources()
                screening_result = screening_engine.screen(question.strip())
                st.session_state["screening_result"] = screening_result.__dict__
                st.session_state["pending_question"] = question.strip()
                st.session_state["question_input"] = question.strip()
                st.session_state["latest_result"] = None
                st.session_state["follow_up_mode"] = "auto_required" if screening_result.needs_follow_up else "optional"
            screening = st.session_state.get("screening_result")

        if screening:
            with st.container(border=True):
                render_screening_card(screening)
                follow_up_mode = st.session_state.get("follow_up_mode")

                if follow_up_mode == "optional":
                    render_follow_up_decision_card(screening)
                    choice_col1, choice_col2 = st.columns(2)
                    with choice_col1:
                        choose_direct = st.button("直接生成回答")
                    with choice_col2:
                        choose_follow_up = st.button("先追问一次")

                    if choose_direct:
                        st.session_state["follow_up_mode"] = "skip_follow_up"
                        st.session_state["latest_result"] = None
                    if choose_follow_up:
                        llm, vectordb, topic_router, query_rewriter, reranker, self_query_parser, screening_engine = load_runtime_resources()
                        pending_question = st.session_state.get("pending_question", question.strip())
                        questions = screening_engine.generate_follow_up_questions(
                            question=pending_question,
                            screening_type=screening.get("screening_type", "unclear"),
                            risk_level=screening.get("risk_level", "low"),
                            should_seek_care=bool(screening.get("should_seek_care", False)),
                            reason=screening.get("reason", "用户主动选择先追问一次。"),
                        )
                        screening["follow_up_questions"] = questions
                        screening["method"] = f"{screening.get('method', 'rule')}+user_follow_up"
                        st.session_state["screening_result"] = screening
                        st.session_state["follow_up_mode"] = "user_requested"
                        st.session_state["latest_result"] = None
                        follow_up_mode = "user_requested"

                if follow_up_mode in {"auto_required", "user_requested"}:
                    render_follow_up_card(screening)
                    follow_up_notes = st.text_area(
                        "请根据追问补充信息",
                        value="",
                        height=150,
                        key="follow_up_notes",
                        placeholder="例如：这种情况持续了2个月，一周大概4-5次，白天会疲劳和注意力不集中……",
                    )
                    submit_follow_up = st.button("补充信息并生成最终回答")
                    if submit_follow_up and follow_up_notes.strip():
                        with st.spinner("正在结合补充信息生成最终回答..."):
                            llm, vectordb, topic_router, query_rewriter, reranker, self_query_parser, screening_engine = load_runtime_resources()
                            analysis_summary = (st.session_state.get("latest_analysis") or {}).get("summary", "")
                            if analysis_summary:
                                augmented_question = build_analysis_augmented_question(st.session_state.get("pending_question", question.strip()), analysis_summary, follow_up_notes)
                            else:
                                augmented_question = build_augmented_question(st.session_state.get("pending_question", question.strip()), follow_up_notes)
                            result = answer_question(augmented_question, llm, vectordb, topic_router, query_rewriter, reranker, self_query_parser)
                            result["screening"] = screening
                            result["augmented_question"] = augmented_question
                            result["analysis"] = st.session_state.get("latest_analysis") or {}
                            st.session_state["latest_result"] = result

                if st.session_state.get("follow_up_mode") == "skip_follow_up" and not st.session_state.get("latest_result"):
                    with st.spinner("正在生成最终回答..."):
                        llm, vectordb, topic_router, query_rewriter, reranker, self_query_parser, screening_engine = load_runtime_resources()
                        base_question = st.session_state.get("pending_question", question.strip())
                        analysis_summary = (st.session_state.get("latest_analysis") or {}).get("summary", "")
                        final_question = build_analysis_augmented_question(base_question, analysis_summary) if analysis_summary else base_question
                        result = answer_question(final_question, llm, vectordb, topic_router, query_rewriter, reranker, self_query_parser)
                        result["screening"] = screening
                        result["analysis"] = st.session_state.get("latest_analysis") or {}
                        st.session_state["latest_result"] = result

        result = st.session_state.get("latest_result")
        if result:
            st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card"><div class="metric-label">识别主题</div><div class="metric-value">{result['topic']}</div></div>
  <div class="metric-card"><div class="metric-label">路由方式</div><div class="metric-value">{result['route_method']}</div></div>
  <div class="metric-card"><div class="metric-label">命中证据</div><div class="metric-value">{len(result['pdf_docs']) + len(result['qa_docs'])} 条</div></div>
  <div class="metric-card"><div class="metric-label">候选文档</div><div class="metric-value">{result.get('rerank', {}).get('candidate_count', 0)} 条</div></div>
  <div class="metric-card"><div class="metric-label">Self-query</div><div class="metric-value">{result.get('self_query', {}).get('method', 'none')}</div></div>
</div>
""", unsafe_allow_html=True)
            with st.container(border=True):
                render_analysis_card(result.get("analysis", {}))
                render_screening_card(result.get("screening", {}))
                render_query_rewrite_card(result)
                render_self_query_card(result)
                render_rerank_card(result)
                render_answer_card(result)

    with right_col:
        with st.container(border=True):
            st.markdown('<div class="section-title">证据与来源</div>', unsafe_allow_html=True)
            result = st.session_state.get("latest_result")
            if not result:
                st.info("提交问题后，这里会显示命中的指南、科普资料和 QA 证据。")
            else:
                st.markdown("### 指南 / 科普资料")
                if result["pdf_docs"]:
                    for idx, doc in enumerate(result["pdf_docs"], start=1):
                        render_reference_card(doc, idx)
                else:
                    st.caption("暂无命中的指南或科普资料")
                st.markdown("### 结构化 QA")
                if result["qa_docs"]:
                    for idx, doc in enumerate(result["qa_docs"], start=1):
                        render_reference_card(doc, idx)
                else:
                    st.caption("暂无命中的 QA 资料")


if __name__ == "__main__":
    main()
