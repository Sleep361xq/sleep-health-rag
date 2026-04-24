import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime

from langchain_core.prompts import PromptTemplate

from config import VECTOR_DB_DIR
from rag_router import answer_question, build_llm, build_reranker, build_vectorstore
from query_rewriter import QueryRewriter
from screening import ScreeningEngine
from self_query import SelfQueryParser
from topic_router import TopicRouter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(BASE_DIR, "data", "test_data", "testset_v1.csv")
DEFAULT_OUT_ROOT = os.path.join(BASE_DIR, "outputs", "evaluations")
PASS = "符合"
PARTIAL = "部分符合"
FAIL = "不符合"
NA = "不适用"
OVERALL_PASS = "通过"
OVERALL_PARTIAL = "基本通过"
OVERALL_FAIL = "不通过"
TOPICS = {"insomnia", "osa", "csa", "sleep_hygiene", "general"}
DOC_TYPES = {"guideline", "education", "other", "qa"}

ANSWER_EVAL_PROMPT = PromptTemplate(
    template="""
你是睡眠健康 RAG 系统的回答质量评估器。

请根据用户问题、系统回答和检索证据，对回答质量进行简要评估。

评估要求：
1. 只输出严格 JSON。
2. 评价维度包括：
   - 是否回答了用户问题
   - 是否与给定证据大体一致
   - 是否存在明显编造或过度结论
   - 是否表达清晰、对用户有帮助
3. 只能输出以下三个标签之一：符合 / 部分符合 / 不符合。
4. 如果回答基本解决问题、没有明显脱离证据或明显编造，则判为“符合”。
5. 如果回答部分有用，但存在轻微遗漏、表达一般、或与证据贴合度一般，则判为“部分符合”。
6. 如果回答明显答非所问、明显脱离证据、存在明显编造或风险性过度结论，则判为“不符合”。
7. 输出格式如下：
{{
  "answer_eval": "符合",
  "reason": "一句简短原因"
}}

用户问题：{question}

系统回答：
{answer}

检索证据摘要：
{evidence}
""",
    input_variables=["question", "answer", "evidence"],
)


def args_parser():
    parser = argparse.ArgumentParser(description="批量运行结构化自动评测")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def clean(v):
    return (v or "").strip()


def norm(v):
    return clean(v).lower()


def none_like(v):
    return norm(v) in {"", "none", "null", "na", "n/a", "不适用"}


def parse_topic_options(v):
    lowered = norm(v)
    return {t for t in TOPICS if t in lowered}


def source_match(expected, actual):
    expected = clean(expected)
    actual = clean(actual)
    if not expected:
        return True
    if not actual:
        return False
    return expected in actual or actual in expected


def fmt_set(values):
    return "|".join(sorted(x for x in values if x))


def answer_preview(text, limit=180):
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit] + " ..."


def build_evidence_preview(pdf_docs, qa_docs, limit=6):
    lines = []
    for doc in pdf_docs[:limit]:
        meta = doc.metadata
        lines.append(
            f"PDF|doc_type={clean(meta.get('doc_type'))}|topic={clean(meta.get('topic'))}|source={clean(meta.get('source'))}|title={clean(meta.get('title') or meta.get('filename'))}"
        )
    for doc in qa_docs[:limit]:
        meta = doc.metadata
        lines.append(
            f"QA|qa_doc_type={clean(meta.get('qa_doc_type'))}|topic={clean(meta.get('topic'))}|source={clean(meta.get('qa_source') or meta.get('source'))}|title={clean(meta.get('title') or meta.get('filename'))}"
        )
    return "\n".join(lines) if lines else "无检索证据"


def eval_answer_quality(llm, question, answer_text, pdf_docs, qa_docs):
    if not clean(answer_text):
        return FAIL, "回答为空"

    try:
        prompt = ANSWER_EVAL_PROMPT.format(
            question=question,
            answer=answer_text,
            evidence=build_evidence_preview(pdf_docs, qa_docs),
        )
        response = llm.invoke(prompt)
        parsed = json.loads(response.content.strip())
        answer_eval = clean(parsed.get("answer_eval"))
        reason = clean(parsed.get("reason")) or "未提供原因"
        if answer_eval not in {PASS, PARTIAL, FAIL}:
            return PARTIAL, f"回答评估输出无效，回退为部分符合：{reason}"
        return answer_eval, reason
    except Exception as exc:
        return PARTIAL, f"回答评估失败，回退为部分符合：{exc}"


def eval_topic(expected, actual):
    opts = parse_topic_options(expected)
    actual = norm(actual)
    if not opts:
        return NA
    if actual in opts:
        return PASS
    if actual == "general":
        return PARTIAL
    return FAIL


def eval_self_query(expected_doc, expected_source, actual_doc, actual_source):
    checks = []
    if not none_like(expected_doc):
        checks.append(norm(expected_doc) == norm(actual_doc))
    if not none_like(expected_source):
        checks.append(source_match(expected_source, actual_source))
    if not checks:
        return NA, NA, NA
    ok = sum(1 for x in checks if x)
    overall = PASS if ok == len(checks) else PARTIAL if ok > 0 else FAIL
    doc_eval = NA if none_like(expected_doc) else (PASS if norm(expected_doc) == norm(actual_doc) else FAIL)
    source_eval = NA if none_like(expected_source) else (PASS if source_match(expected_source, actual_source) else FAIL)
    return overall, doc_eval, source_eval


def eval_follow_up(expected, actual):
    expected = norm(expected)
    actual = norm(actual)
    if expected in {"", "optional"}:
        return PASS
    return PASS if expected == actual else FAIL


def eval_evidence(expected_topic, expected_source, expected_ev, pdf_types, qa_types, ev_topics, ev_sources, qa_count, pdf_count):
    lowered = norm(expected_ev)
    topic_ok = True
    wanted_topics = parse_topic_options(expected_topic)
    if wanted_topics and "general" not in wanted_topics:
        topic_ok = bool(wanted_topics & {norm(x) for x in ev_topics})
    source_ok = True if none_like(expected_source) else any(source_match(expected_source, s) for s in ev_sources)
    type_checks = []
    if "qa_guideline" in lowered:
        type_checks.append("guideline" in qa_types)
    if "qa_education" in lowered:
        type_checks.append("education" in qa_types)
    if "qa_other" in lowered:
        type_checks.append("other" in qa_types)
    if "guideline" in lowered:
        type_checks.append("guideline" in pdf_types)
    if "education" in lowered:
        type_checks.append("education" in pdf_types)
    if "other" in lowered and "qa_other" not in lowered:
        type_checks.append("other" in pdf_types)
    if "mixed evidence" in lowered:
        type_checks.append(qa_count > 0 and pdf_count > 0)
    if "qa" in lowered and all(x not in lowered for x in ["qa_guideline", "qa_education", "qa_other"]):
        type_checks.append(qa_count > 0)
    type_ok = True if not type_checks else any(type_checks)
    score = sum(1 for x in [topic_ok, source_ok, type_ok] if x)
    return PASS if score == 3 else PARTIAL if score >= 2 else FAIL


def overall_eval(topic_eval, sq_eval, follow_eval, evidence_eval, answer_eval, run_status):
    if run_status != "success":
        return OVERALL_FAIL
    vals = [topic_eval, sq_eval, follow_eval, evidence_eval, answer_eval]
    fail_count = sum(1 for v in vals if v == FAIL)
    partial_count = sum(1 for v in vals if v == PARTIAL)
    if fail_count == 0 and partial_count <= 1:
        return OVERALL_PASS
    if fail_count <= 1:
        return OVERALL_PARTIAL
    return OVERALL_FAIL


def load_cases(path, limit):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if clean(r.get("question"))]
    return rows[:limit] if limit else rows


def build_resources():
    llm = build_llm()
    vectordb = build_vectorstore()
    return (
        llm,
        vectordb,
        TopicRouter(llm=llm),
        QueryRewriter(llm=llm),
        build_reranker(),
        SelfQueryParser(llm=llm),
        ScreeningEngine(llm=llm),
    )


def evaluate_case(row, llm, vectordb, topic_router, query_rewriter, reranker, self_query_parser, screening_engine):
    question = clean(row.get("question"))
    result = dict(row)
    try:
        screening = screening_engine.screen(question)
        answer = answer_question(
            question=question,
            llm=llm,
            vectordb=vectordb,
            topic_router=topic_router,
            query_rewriter=query_rewriter,
            reranker=reranker,
            self_query_parser=self_query_parser,
        )
        sq = answer.get("self_query", {})
        qa_docs = answer.get("qa_docs", [])
        pdf_docs = answer.get("pdf_docs", [])
        pdf_types = {d.metadata.get("doc_type", "") for d in pdf_docs if d.metadata.get("doc_type")}
        qa_types = {d.metadata.get("qa_doc_type", "") for d in qa_docs if d.metadata.get("qa_doc_type")}
        ev_topics = {d.metadata.get("topic", "") for d in qa_docs + pdf_docs if d.metadata.get("topic")}
        ev_sources = set()
        for d in pdf_docs:
            if d.metadata.get("source"):
                ev_sources.add(d.metadata.get("source"))
        for d in qa_docs:
            src = d.metadata.get("qa_source") or d.metadata.get("source")
            if src:
                ev_sources.add(src)

        actual_topic = clean(answer.get("topic"))
        actual_doc = clean(sq.get("doc_type")) or "none"
        actual_source = clean(sq.get("source")) or "none"
        actual_follow = "yes" if screening.needs_follow_up else "no"
        actual_evidence = f"pdf:{fmt_set(pdf_types) or 'none'}; qa:{fmt_set(qa_types) or 'none'}; topics:{fmt_set(ev_topics) or 'none'}"

        topic_eval = eval_topic(row.get("expected_topic", ""), actual_topic)
        sq_eval, doc_eval, source_eval = eval_self_query(
            row.get("expected_doc_type", ""),
            row.get("expected_source", ""),
            actual_doc,
            actual_source,
        )
        follow_eval = eval_follow_up(row.get("expected_follow_up", ""), actual_follow)
        evidence_eval = eval_evidence(
            row.get("expected_topic", ""),
            row.get("expected_source", ""),
            row.get("expected_evidence_type", ""),
            pdf_types,
            qa_types,
            ev_topics,
            ev_sources,
            len(qa_docs),
            len(pdf_docs),
        )
        answer_eval, answer_eval_reason = eval_answer_quality(
            llm=llm,
            question=question,
            answer_text=answer.get("answer", ""),
            pdf_docs=pdf_docs,
            qa_docs=qa_docs,
        )
        result.update({
            "actual_topic": actual_topic,
            "actual_doc_type": actual_doc,
            "actual_source": actual_source,
            "actual_follow_up": actual_follow,
            "actual_evidence_type": actual_evidence,
            "topic_eval": topic_eval,
            "self_query_eval": sq_eval,
            "follow_up_eval": follow_eval,
            "evidence_eval": evidence_eval,
            "answer_eval": answer_eval,
            "overall_eval": overall_eval(topic_eval, sq_eval, follow_eval, evidence_eval, answer_eval, "success"),
            "run_status": "success",
            "error": "",
            "route_method": clean(answer.get("route_method")),
            "route_reason": clean(answer.get("route_reason")),
            "self_query_method": clean(sq.get("method")),
            "self_query_reason": clean(sq.get("reason")),
            "answer_eval_reason": answer_eval_reason,
            "doc_type_eval": doc_eval,
            "source_eval": source_eval,
            "screening_type": screening.screening_type,
            "risk_level": screening.risk_level,
            "should_seek_care": str(screening.should_seek_care).lower(),
            "screening_method": screening.method,
            "screening_reason": screening.reason,
            "follow_up_questions": " | ".join(screening.follow_up_questions),
            "query_is_complex": str(answer.get("query_rewrite", {}).get("is_complex")),
            "query_rewrite_method": clean(answer.get("query_rewrite", {}).get("rewrite_method")),
            "rewritten_queries": " | ".join(answer.get("query_rewrite", {}).get("rewritten_queries", [])),
            "pdf_doc_types": fmt_set(pdf_types),
            "qa_doc_types": fmt_set(qa_types),
            "evidence_topics": fmt_set(ev_topics),
            "evidence_sources": fmt_set(ev_sources),
            "qa_count": str(len(qa_docs)),
            "pdf_count": str(len(pdf_docs)),
            "rerank_candidate_count": str(answer.get("rerank", {}).get("candidate_count", "")),
            "answer_preview": answer_preview(answer.get("answer", "")),
        })
        return result
    except Exception as exc:
        result.update({
            "actual_topic": "",
            "actual_doc_type": "",
            "actual_source": "",
            "actual_follow_up": "",
            "actual_evidence_type": "",
            "topic_eval": FAIL,
            "self_query_eval": FAIL,
            "follow_up_eval": FAIL,
            "evidence_eval": FAIL,
            "answer_eval": NA,
            "overall_eval": OVERALL_FAIL,
            "run_status": "error",
            "error": str(exc),
            "answer_preview": "",
        })
        return result


def counter(results, field):
    return dict(Counter(r.get(field, "") for r in results))


def build_summary(results, input_csv, output_dir):
    total = len(results)
    success = sum(1 for r in results if r.get("run_status") == "success")
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_csv": input_csv,
        "output_dir": output_dir,
        "total_cases": total,
        "success_cases": success,
        "error_cases": total - success,
        "overall_eval_counts": counter(results, "overall_eval"),
        "topic_eval_counts": counter(results, "topic_eval"),
        "self_query_eval_counts": counter(results, "self_query_eval"),
        "follow_up_eval_counts": counter(results, "follow_up_eval"),
        "evidence_eval_counts": counter(results, "evidence_eval"),
        "answer_eval_counts": counter(results, "answer_eval"),
        "run_status_counts": counter(results, "run_status"),
        "failed_case_ids": [r.get("id", "") for r in results if r.get("overall_eval") == OVERALL_FAIL],
        "error_case_ids": [r.get("id", "") for r in results if r.get("run_status") == "error"],
    }


def ensure_out_dir(output_dir):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(DEFAULT_OUT_ROOT, f"run_{ts}")
    os.makedirs(path, exist_ok=True)
    return path


def write_detail_csv(path, results):
    if not results:
        return
    front = [
        "id", "question", "category", "difficulty", "expected_topic", "expected_doc_type", "expected_source",
        "expected_follow_up", "expected_evidence_type", "expected_behavior", "actual_topic", "actual_doc_type",
        "actual_source", "actual_follow_up", "actual_evidence_type", "topic_eval", "self_query_eval",
        "follow_up_eval", "evidence_eval", "answer_eval", "overall_eval", "run_status", "error", "notes"
    ]
    fields = []
    for row in results:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    ordered = [x for x in front if x in fields] + [x for x in fields if x not in front]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(results)


def count_value(mapping, key):
    return mapping.get(key, 0)


def format_counter_line(title, counts, ordered_keys):
    parts = [f"{key} {counts.get(key, 0)}" for key in ordered_keys if counts.get(key, 0)]
    suffix = "，".join(parts) if parts else "无"
    return f"- {title}：{suffix}"


def build_conclusion_lines(summary):
    topic_counts = summary.get("topic_eval_counts", {})
    follow_counts = summary.get("follow_up_eval_counts", {})
    sq_counts = summary.get("self_query_eval_counts", {})
    answer_counts = summary.get("answer_eval_counts", {})
    evidence_counts = summary.get("evidence_eval_counts", {})
    total = summary.get("total_cases", 0)

    lines = []
    if count_value(topic_counts, PASS) == total and total > 0:
        lines.append("- Topic routing 已稳定，本次所有样例均路由正确。")
    elif count_value(topic_counts, FAIL) == 0:
        lines.append("- Topic routing 整体稳定，没有出现明显错误路由。")
    else:
        lines.append("- Topic routing 仍存在错误路由，建议继续关注主题分类规则。")

    if count_value(evidence_counts, FAIL) == 0:
        lines.append("- Evidence retrieval 整体表现稳定，没有出现明显证据召回失败。")
    else:
        lines.append("- Evidence retrieval 仍有部分失败样例，值得继续排查检索链路。")

    if count_value(follow_counts, FAIL) > 0:
        lines.append("- Follow-up decision 仍是当前主要薄弱环节，存在较多与预期不一致的样例。")
    else:
        lines.append("- Follow-up decision 与预期一致，当前无需优先优化。")

    if count_value(sq_counts, FAIL) > 0 or count_value(sq_counts, PARTIAL) > 0:
        lines.append("- Self-query 已具备可用性，但在隐式约束识别上仍有继续优化空间。")
    else:
        lines.append("- Self-query 表现稳定，metadata 约束识别基本符合预期。")

    if count_value(answer_counts, FAIL) > 0:
        lines.append("- 回答质量评估显示仍存在明显问题样例，建议结合 answer_eval_reason 做进一步复盘。")
    elif count_value(answer_counts, PARTIAL) > 0:
        lines.append("- 回答质量整体可用，但仍有部分样例在完整性或贴证据程度上有提升空间。")
    else:
        lines.append("- 回答质量整体稳定，回答基本能够贴合问题与证据。")

    return lines


def build_focus_case_sections(results):
    failed = [r for r in results if r.get("overall_eval") == OVERALL_FAIL]
    borderline = [
        r for r in results
        if r.get("overall_eval") == OVERALL_PARTIAL
        and any(r.get(field) in {FAIL, PARTIAL} for field in ["topic_eval", "self_query_eval", "follow_up_eval", "evidence_eval"])
    ]

    lines = []
    lines.append("## 重点关注样例")

    if failed:
        lines.append("")
        lines.append("### 不通过")
        for row in failed[:10]:
            lines.extend(format_case_block(row))
    else:
        lines.append("")
        lines.append("### 不通过")
        lines.append("- 无")

    lines.append("")
    lines.append("### 基本通过但值得关注")
    if borderline:
        for row in borderline[:10]:
            lines.extend(format_case_block(row))
    else:
        lines.append("- 无")

    return lines


def format_case_block(row):
    issue_fields = [
        ("Topic", row.get("topic_eval", "")),
        ("Self-query", row.get("self_query_eval", "")),
        ("Follow-up", row.get("follow_up_eval", "")),
        ("Evidence", row.get("evidence_eval", "")),
        ("Answer", row.get("answer_eval", "")),
    ]
    issues = [f"{name}：{value}" for name, value in issue_fields if value in {FAIL, PARTIAL}]
    behavior = clean(row.get("expected_behavior"))
    answer_reason = clean(row.get("answer_eval_reason"))
    lines = [f"- {clean(row.get('id'))}｜{clean(row.get('question'))}"]
    if issues:
        lines.append(f"  - 问题维度：{'；'.join(issues)}")
    if answer_reason:
        lines.append(f"  - 回答评估：{answer_reason}")
    if behavior:
        lines.append(f"  - 预期：{behavior}")
    return lines


def build_category_lines(results):
    category_map = {}
    for row in results:
        category = clean(row.get("category")) or "uncategorized"
        bucket = category_map.setdefault(category, [])
        bucket.append(row)

    lines = ["## 按类别统计"]
    for category in sorted(category_map):
        rows = category_map[category]
        overall_counts = Counter(r.get("overall_eval", "") for r in rows)
        parts = [
            f"{len(rows)} 条",
            f"通过 {overall_counts.get(OVERALL_PASS, 0)}",
            f"基本通过 {overall_counts.get(OVERALL_PARTIAL, 0)}",
            f"不通过 {overall_counts.get(OVERALL_FAIL, 0)}",
        ]
        lines.append(f"- {category}：{'，'.join(parts)}")
    return lines


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_previous_summary(current_summary_path, current_summary):
    current_dir = os.path.dirname(current_summary_path)
    target_input = os.path.basename(current_summary.get("input_csv", ""))
    candidates = []

    for entry in os.scandir(DEFAULT_OUT_ROOT):
        if not entry.is_dir() or os.path.abspath(entry.path) == os.path.abspath(current_dir):
            continue
        summary_path = os.path.join(entry.path, "test_results_summary.json")
        if not os.path.exists(summary_path):
            continue
        try:
            summary = load_json(summary_path)
        except Exception:
            continue
        same_input = os.path.basename(summary.get("input_csv", "")) == target_input
        generated_at = summary.get("generated_at", "")
        candidates.append((same_input, generated_at, summary_path, summary))

    if not candidates:
        return None, None

    same_input_candidates = [item for item in candidates if item[0]]
    pool = same_input_candidates or candidates
    pool.sort(key=lambda item: item[1], reverse=True)
    _, _, path, summary = pool[0]
    return path, summary


def get_count_delta(current_counts, previous_counts, key):
    return current_counts.get(key, 0) - previous_counts.get(key, 0)


def format_delta(value):
    if value > 0:
        return f"+{value}"
    return str(value)


def build_comparison_lines(current_summary, previous_summary_path, previous_summary):
    lines = ["## 与上一次结果对比"]
    if not previous_summary:
        lines.append("- 无可对比的历史结果")
        return lines

    lines.append(f"- 对比基线：`{previous_summary_path}`")

    metrics = [
        ("通过", "overall_eval_counts", OVERALL_PASS),
        ("基本通过", "overall_eval_counts", OVERALL_PARTIAL),
        ("不通过", "overall_eval_counts", OVERALL_FAIL),
        ("Topic-符合", "topic_eval_counts", PASS),
        ("Self-query-符合", "self_query_eval_counts", PASS),
        ("Follow-up-符合", "follow_up_eval_counts", PASS),
        ("Evidence-符合", "evidence_eval_counts", PASS),
        ("Answer-符合", "answer_eval_counts", PASS),
    ]

    for label, field, key in metrics:
        current_counts = current_summary.get(field, {})
        previous_counts = previous_summary.get(field, {})
        current_value = current_counts.get(key, 0)
        previous_value = previous_counts.get(key, 0)
        delta = get_count_delta(current_counts, previous_counts, key)
        lines.append(f"- {label}：{previous_value} → {current_value}（{format_delta(delta)}）")

    return lines


def write_markdown_report(path, summary, results, detail_path, summary_path):
    previous_summary_path, previous_summary = discover_previous_summary(summary_path, summary)
    lines = [
        "# 睡眠健康 RAG 自动评测报告",
        "",
        f"- 评测时间：{summary.get('generated_at', '')}",
        f"- 测试集：{os.path.basename(summary.get('input_csv', ''))}",
        f"- 总样例数：{summary.get('total_cases', 0)}",
        f"- 成功运行：{summary.get('success_cases', 0)}",
        f"- 报错：{summary.get('error_cases', 0)}",
        "",
        "## Overall",
        f"- 通过：{count_value(summary.get('overall_eval_counts', {}), OVERALL_PASS)}",
        f"- 基本通过：{count_value(summary.get('overall_eval_counts', {}), OVERALL_PARTIAL)}",
        f"- 不通过：{count_value(summary.get('overall_eval_counts', {}), OVERALL_FAIL)}",
        "",
        "## 关键指标摘要",
        format_counter_line("Topic", summary.get("topic_eval_counts", {}), [PASS, PARTIAL, FAIL, NA]),
        format_counter_line("Self-query", summary.get("self_query_eval_counts", {}), [PASS, PARTIAL, FAIL, NA]),
        format_counter_line("Follow-up", summary.get("follow_up_eval_counts", {}), [PASS, PARTIAL, FAIL, NA]),
        format_counter_line("Evidence", summary.get("evidence_eval_counts", {}), [PASS, PARTIAL, FAIL, NA]),
        format_counter_line("Answer", summary.get("answer_eval_counts", {}), [PASS, PARTIAL, FAIL, NA]),
        "",
        *build_comparison_lines(summary, previous_summary_path, previous_summary),
        "",
        "## 本次结论",
        *build_conclusion_lines(summary),
        "",
        *build_focus_case_sections(results),
        "",
        *build_category_lines(results),
        "",
        "## 产出文件",
        f"- 明细结果：`{detail_path}`",
        f"- 汇总结果：`{summary_path}`",
        f"- 可读报告：`{path}`",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = args_parser()
    input_csv = os.path.abspath(args.input)
    output_dir = ensure_out_dir(args.output_dir)
    cases = load_cases(input_csv, args.limit)
    llm, vectordb, topic_router, query_rewriter, reranker, self_query_parser, screening_engine = build_resources()
    results = []
    for idx, row in enumerate(cases, start=1):
        case_id = row.get("id", f"case_{idx}")
        print(f"[{idx}/{len(cases)}] Running {case_id}: {clean(row.get('question'))}")
        results.append(evaluate_case(row, llm, vectordb, topic_router, query_rewriter, reranker, self_query_parser, screening_engine))
    detail_path = os.path.join(output_dir, "test_results_detail.csv")
    summary_path = os.path.join(output_dir, "test_results_summary.json")
    report_path = os.path.join(output_dir, "evaluation_report.md")
    write_detail_csv(detail_path, results)
    summary = build_summary(results, input_csv, output_dir)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_markdown_report(report_path, summary, results, detail_path, summary_path)
    print("\n评测完成：")
    print(f"- detail csv: {detail_path}")
    print(f"- summary json: {summary_path}")
    print(f"- readable report: {report_path}")
    print(f"- total cases: {summary['total_cases']}")
    print(f"- success cases: {summary['success_cases']}")
    print(f"- error cases: {summary['error_cases']}")
    print(f"- overall eval counts: {summary['overall_eval_counts']}")


if __name__ == "__main__":
    main()
