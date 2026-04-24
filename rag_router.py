import os
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from config import (
    EMBEDDING_MODEL,
    FINAL_GUIDELINE_MIN,
    FINAL_PDF_LIMIT,
    FINAL_QA_LIMIT,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    MIN_FILTER_RESULTS,
    PDF_DOC_TYPES,
    VECTOR_DB_DIR,
)
from query_rewriter import QueryRewriteResult, QueryRewriter
from reranker import Reranker
from self_query import SelfQueryParser, SelfQueryResult
from topic_router import TopicRouteResult, TopicRouter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = str(VECTOR_DB_DIR)

PROMPT = PromptTemplate(
    template="""
你是专业的睡眠健康助手，只能依据给定参考资料回答，禁止编造。
1. 先给出【核心结论】。
2. 再给出【详细说明】，按要点分条表述。
3. 再给出【建议怎么做】。
4. 如果问题涉及风险或可能的睡眠障碍，给出【何时建议就医】。
5. 最后单独列出【参考来源】。
6. 如果资料不足以支持明确结论，要明确写“现有资料未明确提及”。
7. 不要给出确诊结论，不要替代医生诊断，不要编造药物处方方案。
8. QA资料引用 `[QA来源：xxx]`；文档资料引用 `[标题：xxx | 来源：xxx | 章节：xxx]`。
9. 如果提供了检索子问题，请综合这些子问题检索到的资料统一回答原始问题。
10. 如果原始问题里包含“个体睡眠数据分析结果”或工具推理结果，请先解释这些结果代表什么，再结合参考资料说明其可能意义与边界。
11. 结尾必须加一句：本回答仅供健康参考，不替代临床诊断。

原始问题：{question}
检索策略：{rewrite_summary}
主问题主题：{topic}
主题识别方式：{route_method}
主题识别说明：{route_reason}

参考资料：
{context}
""",
    input_variables=["question", "rewrite_summary", "topic", "route_method", "route_reason", "context"],
)


def get_doc_title(m: Dict[str, str]) -> str:
    return m.get("title", m.get("filename", "未知文档"))


def get_doc_source(m: Dict[str, str]) -> str:
    return m.get("source", m.get("filename", "未知来源"))


def get_doc_section(m: Dict[str, str]) -> str:
    return m.get("section_path", "")


def get_doc_rerank_score(m: Dict[str, str]) -> str:
    return m.get("rerank_score", "")


def format_doc_label(doc: Document) -> str:
    m = doc.metadata
    if m.get("doc_type") == "qa":
        qa_type = m.get("qa_doc_type")
        qa_type_label = f" | QA类型：{qa_type}" if qa_type else ""
        return f"[QA来源：{m.get('qa_source', get_doc_source(m))}{qa_type_label}]"
    parts = [f"标题：{get_doc_title(m)}", f"来源：{get_doc_source(m)}"]
    if get_doc_section(m):
        parts.append(f"章节：{get_doc_section(m)}")
    return f"[{' | '.join(parts)}]"


def format_context_block(title: str, docs: List[Document]) -> str:
    if not docs:
        return f"### {title}\n暂无检索结果\n"
    lines = [f"### {title}"]
    for i, doc in enumerate(docs, 1):
        content = doc.page_content.strip().replace("\n", " ")
        lines.append(f"{i}. {format_doc_label(doc)} | topic={doc.metadata.get('topic', 'unknown')}\n{content}")
    return "\n\n".join(lines)


def unique_documents(documents: List[Document]) -> List[Document]:
    seen, out = set(), []
    for doc in documents:
        key = (doc.metadata.get("filename"), doc.metadata.get("section_path"), doc.metadata.get("qa_source"), doc.page_content)
        if key not in seen:
            seen.add(key)
            out.append(doc)
    return out


def build_metadata_filter(doc_type: Optional[str] = None, topic: Optional[str] = None, source: Optional[str] = None, qa_doc_type: Optional[str] = None) -> Dict[str, object]:
    conditions: List[Dict[str, str]] = []
    if doc_type:
        conditions.append({"doc_type": doc_type})
    if topic:
        conditions.append({"topic": topic})
    if source:
        conditions.append({"source": source})
    if qa_doc_type:
        conditions.append({"qa_doc_type": qa_doc_type})
    if not conditions:
        return {}
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def do_search(vectordb: Chroma, question: str, k: int, metadata_filter: Dict[str, object]) -> List[Document]:
    return vectordb.similarity_search(question, k=k, filter=metadata_filter) if metadata_filter else vectordb.similarity_search(question, k=k)


def search_documents(vectordb: Chroma, question: str, doc_type: str, primary_topic: str, limit: int, allow_general_fallback: bool, sq: SelfQueryResult) -> List[Document]:
    effective_doc_type = sq.doc_type or doc_type
    effective_topic = sq.topic or primary_topic
    qa_doc_type = sq.doc_type if doc_type == "qa" and sq.doc_type in PDF_DOC_TYPES else None
    base_doc_type = "qa" if doc_type == "qa" else effective_doc_type
    fallback_doc_type = "qa" if doc_type == "qa" else (effective_doc_type or doc_type)

    filters = [
        build_metadata_filter(base_doc_type, effective_topic, None, qa_doc_type),
        build_metadata_filter(fallback_doc_type, primary_topic, None, qa_doc_type),
        build_metadata_filter(doc_type, None, None, qa_doc_type),
    ]
    docs: List[Document] = []
    for metadata_filter in filters:
        docs = unique_documents(do_search(vectordb, question, limit, metadata_filter))
        if len(docs) >= MIN_FILTER_RESULTS:
            break

    if sq.source:
        docs.extend(
            do_search(
                vectordb,
                question,
                max(2, limit // 2 + 1),
                build_metadata_filter(base_doc_type, effective_topic, sq.source, qa_doc_type),
            )
        )

    if allow_general_fallback and effective_topic != "general":
        docs.extend(do_search(vectordb, question, max(3, limit // 2 + 1), build_metadata_filter(fallback_doc_type, "general", None, qa_doc_type)))
        if sq.source:
            docs.extend(
                do_search(
                    vectordb,
                    question,
                    max(2, limit // 2),
                    build_metadata_filter(fallback_doc_type, "general", sq.source, qa_doc_type),
                )
            )

    if (primary_topic == "general" or not docs) and not any([sq.doc_type, sq.source, sq.topic]):
        docs.extend(do_search(vectordb, question, limit, build_metadata_filter(doc_type)))
    return unique_documents(docs)


def retrieve_documents_for_topic(vectordb: Chroma, question: str, topic: str, sq: SelfQueryResult) -> Tuple[List[Document], List[Document]]:
    qa_docs = search_documents(vectordb, question, "qa", topic, 5, True, sq)
    pdf_docs: List[Document] = []
    for doc_type in PDF_DOC_TYPES:
        pdf_docs.extend(search_documents(vectordb, question, doc_type, topic, 6 if doc_type == "guideline" else 4, doc_type != "guideline", sq))
    return unique_documents(qa_docs), unique_documents(pdf_docs)


def select_final_documents(reranked_docs: List[Document]) -> Tuple[List[Document], List[Document]]:
    pdf_candidates = [d for d in reranked_docs if d.metadata.get("doc_type") != "qa"]
    qa_candidates = [d for d in reranked_docs if d.metadata.get("doc_type") == "qa"]
    guideline_docs = [d for d in pdf_candidates if d.metadata.get("doc_type") == "guideline"]
    other_docs = [d for d in pdf_candidates if d.metadata.get("doc_type") != "guideline"]
    selected_pdf_docs = guideline_docs[:FINAL_GUIDELINE_MIN]
    used = {(d.metadata.get("filename"), d.metadata.get("section_path"), d.page_content) for d in selected_pdf_docs}
    for doc in guideline_docs[FINAL_GUIDELINE_MIN:] + other_docs:
        if len(selected_pdf_docs) >= FINAL_PDF_LIMIT:
            break
        key = (doc.metadata.get("filename"), doc.metadata.get("section_path"), doc.page_content)
        if key not in used:
            used.add(key)
            selected_pdf_docs.append(doc)
    return qa_candidates[:FINAL_QA_LIMIT], selected_pdf_docs


def retrieve_documents(vectordb: Chroma, topic_router: TopicRouter, reranker: Reranker, question: str, rewrite_result: QueryRewriteResult, sq: SelfQueryResult):
    all_candidates: List[Document] = []
    topic_results: List[TopicRouteResult] = []
    for query in rewrite_result.rewritten_queries:
        topic_result = topic_router.route(query)
        topic_results.append(topic_result)
        qa_docs, pdf_docs = retrieve_documents_for_topic(vectordb, query, topic_result.topic, sq)
        all_candidates.extend(qa_docs)
        all_candidates.extend(pdf_docs)
    reranked_docs = reranker.rerank(question, unique_documents(all_candidates))
    qa_docs, pdf_docs = select_final_documents(reranked_docs)
    return qa_docs, pdf_docs, topic_results, reranked_docs


def build_rewrite_summary(rewrite_result: QueryRewriteResult, topic_results: List[TopicRouteResult], sq: SelfQueryResult) -> str:
    if not rewrite_result.is_complex:
        topic = topic_results[0].topic if topic_results else "unknown"
        base = f"单问题检索；复杂度判断：{rewrite_result.complexity_reason}；识别主题：{topic}"
    else:
        lines = [f"复杂问题多查询检索；复杂度判断：{rewrite_result.complexity_reason}", f"查询拆分方式：{rewrite_result.rewrite_method}；拆分说明：{rewrite_result.rewrite_reason}", "检索子问题："]
        lines.extend([f"- 子问题：{q}（主题：{t.topic}，路由：{t.method}）" for q, t in zip(rewrite_result.rewritten_queries, topic_results)])
        base = "\n".join(lines)
    filters = []
    if sq.doc_type:
        filters.append(f"doc_type={sq.doc_type}")
    if sq.source:
        filters.append(f"source={sq.source}")
    if sq.topic:
        filters.append(f"topic={sq.topic}")
    return f"{base}\nSelf-query 约束：{'；'.join(filters) if filters else '无显式 metadata 过滤'}"


def build_context(question: str, vectordb: Chroma, topic_router: TopicRouter, query_rewriter: QueryRewriter, reranker: Reranker, self_query_parser: SelfQueryParser):
    rewrite_result = query_rewriter.rewrite(question)
    sq = self_query_parser.parse(question)
    qa_docs, pdf_docs, topic_results, reranked_docs = retrieve_documents(vectordb, topic_router, reranker, question, rewrite_result, sq)
    context = "\n\n".join([format_context_block("指南/科普资料", pdf_docs), format_context_block("结构化问答资料", qa_docs)])
    return context, qa_docs, pdf_docs, rewrite_result, topic_results, reranked_docs, sq


def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
    )


def build_vectorstore() -> Chroma:
    return Chroma(persist_directory=DB_DIR, embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))


def build_reranker() -> Reranker:
    return Reranker()


def answer_question(question: str, llm: ChatOpenAI, vectordb: Chroma, topic_router: TopicRouter, query_rewriter: QueryRewriter, reranker: Reranker, self_query_parser: SelfQueryParser) -> Dict[str, object]:
    context, qa_docs, pdf_docs, rewrite_result, topic_results, reranked_docs, sq = build_context(question, vectordb, topic_router, query_rewriter, reranker, self_query_parser)
    main_topic_result = topic_results[0] if topic_results else topic_router.route(question)
    prompt = PROMPT.format(question=question, rewrite_summary=build_rewrite_summary(rewrite_result, topic_results, sq), topic=main_topic_result.topic, route_method=main_topic_result.method, route_reason=main_topic_result.reason, context=context)
    response = llm.invoke(prompt)
    return {
        "topic": main_topic_result.topic,
        "route_method": main_topic_result.method,
        "route_reason": main_topic_result.reason,
        "rule_scores": main_topic_result.rule_scores,
        "answer": response.content,
        "qa_docs": qa_docs,
        "pdf_docs": pdf_docs,
        "query_rewrite": {"is_complex": rewrite_result.is_complex, "complexity_method": rewrite_result.complexity_method, "complexity_reason": rewrite_result.complexity_reason, "rewrite_method": rewrite_result.rewrite_method, "rewrite_reason": rewrite_result.rewrite_reason, "rewritten_queries": rewrite_result.rewritten_queries},
        "self_query": {"doc_type": sq.doc_type, "source": sq.source, "topic": sq.topic, "method": sq.method, "reason": sq.reason},
        "sub_query_topics": [{"query": q, "topic": t.topic, "route_method": t.method, "route_reason": t.reason} for q, t in zip(rewrite_result.rewritten_queries, topic_results)],
        "rerank": {"enabled": True, "candidate_count": len(reranked_docs), "final_pdf_count": len(pdf_docs), "final_qa_count": len(qa_docs), "guideline_min": FINAL_GUIDELINE_MIN},
    }


def print_references(title: str, docs: List[Document]):
    print(f"\n📎 {title}")
    if not docs:
        print("- 暂无")
        return
    for doc in docs:
        m = doc.metadata
        suffix = f" | rerank={get_doc_rerank_score(m)}" if get_doc_rerank_score(m) else ""
        if m.get("doc_type") == "qa":
            qa_doc_type = m.get("qa_doc_type", "unknown")
            print(f"- QA | qa_doc_type={qa_doc_type} | topic={m.get('topic', 'unknown')} | title={get_doc_title(m)} | source={m.get('qa_source', get_doc_source(m))}{suffix}")
        elif get_doc_section(m):
            print(f"- {m.get('doc_type', 'unknown')} | topic={m.get('topic', 'unknown')} | title={get_doc_title(m)} | source={get_doc_source(m)} | section={get_doc_section(m)}{suffix}")
        else:
            print(f"- {m.get('doc_type', 'unknown')} | topic={m.get('topic', 'unknown')} | title={get_doc_title(m)} | source={get_doc_source(m)}{suffix}")


def print_query_rewrite(result: Dict[str, object]):
    rewrite = result.get("query_rewrite", {})
    print("\n🔍 查询重构")
    print(f"- 是否复杂问题：{rewrite.get('is_complex')}")
    print(f"- 复杂度判断方式：{rewrite.get('complexity_method')}")
    print(f"- 复杂度判断说明：{rewrite.get('complexity_reason')}")
    print(f"- 改写方式：{rewrite.get('rewrite_method')}")
    print(f"- 改写说明：{rewrite.get('rewrite_reason')}")
    for item in result.get("sub_query_topics", []):
        print(f"  · 子问题：{item['query']} | 主题：{item['topic']} | 路由方式：{item['route_method']} | 路由说明：{item['route_reason']}")


def print_self_query_info(result: Dict[str, object]):
    info = result.get("self_query", {})
    print("\n🧠 Self-query")
    print(f"- doc_type：{info.get('doc_type')}")
    print(f"- source：{info.get('source')}")
    print(f"- topic：{info.get('topic')}")
    print(f"- 解析方式：{info.get('method')}")
    print(f"- 解析说明：{info.get('reason')}")


def print_rerank_info(result: Dict[str, object]):
    rerank = result.get("rerank", {})
    print("\n🎯 重排序")
    print(f"- 是否启用：{rerank.get('enabled')}")
    print(f"- 候选文档数：{rerank.get('candidate_count')}")
    print(f"- 最终指南/科普数：{rerank.get('final_pdf_count')}")
    print(f"- 最终QA数：{rerank.get('final_qa_count')}")
    print(f"- 最少保留指南数：{rerank.get('guideline_min')}")


def main():
    llm = build_llm()
    vectordb = build_vectorstore()
    topic_router = TopicRouter(llm=llm)
    query_rewriter = QueryRewriter(llm=llm)
    reranker = build_reranker()
    self_query_parser = SelfQueryParser(llm=llm)
    print("💤 睡眠健康RAG系统（规则 + LLM 主题路由版）已启动（输入 exit 退出）")
    while True:
        question = input("\n请输入你的问题：").strip()
        if not question:
            continue
        if question.lower() == "exit":
            break
        result = answer_question(question, llm, vectordb, topic_router, query_rewriter, reranker, self_query_parser)
        print(f"\n🎯 识别主题：{result['topic']}")
        print(f"🧭 路由方式：{result['route_method']}")
        print(f"📝 路由说明：{result['route_reason']}")
        print(f"📊 规则得分：{result['rule_scores']}")
        print_query_rewrite(result)
        print_self_query_info(result)
        print_rerank_info(result)
        print("\n📝 回答：\n")
        print(result["answer"])
        print_references("命中的指南/科普资料", result["pdf_docs"])
        print_references("命中的QA资料", result["qa_docs"])


if __name__ == "__main__":
    main()
