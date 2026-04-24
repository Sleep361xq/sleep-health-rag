import csv
import warnings
from typing import Dict, List

warnings.filterwarnings("ignore")

from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import (
    DATA_DIR,
    EMBEDDING_MODEL,
    MARKDOWN_HEADERS_TO_SPLIT_ON,
    QA_DIR,
    QA_RAW_METADATA_FIELDS,
    REGISTRY_PATH,
    SUPPORTED_MD_DIRS,
    TEXT_ENCODINGS,
    VECTOR_DB_DIR,
)

BASE_DIR = DATA_DIR.parent


def clean_metadata(metadata: Dict[str, str]) -> Dict[str, str]:
    return {
        key: value
        for key, value in metadata.items()
        if value not in (None, "")
    }


def detect_text_encoding(file_path: str) -> str:
    for encoding in TEXT_ENCODINGS:
        try:
            with open(file_path, "r", encoding=encoding, newline="") as file:
                sample = file.read(4096)
                if sample is not None:
                    return encoding
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"无法识别文件编码：{file_path}",
    )


def load_metadata_registry() -> Dict[str, Dict[str, str]]:
    registry: Dict[str, Dict[str, str]] = {}
    encoding = detect_text_encoding(REGISTRY_PATH)

    with open(REGISTRY_PATH, "r", encoding=encoding, newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            filename = (row.get("filename") or "").strip()
            if not filename:
                continue

            registry[filename] = clean_metadata(
                {key: (value or "").strip() for key, value in row.items()}
            )

    return registry


def get_registry_metadata(
    filename: str,
    registry: Dict[str, Dict[str, str]],
    default_doc_type: str,
) -> Dict[str, str]:
    metadata = dict(registry.get(filename, {}))
    fallback_title = os.path.splitext(filename)[0]
    metadata.setdefault("filename", filename)
    metadata.setdefault("title", fallback_title)
    metadata.setdefault("doc_type", default_doc_type)
    return clean_metadata(metadata)


def get_csv_metadata_columns(csv_path: str) -> List[str]:
    encoding = detect_text_encoding(csv_path)

    with open(csv_path, "r", encoding=encoding, newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []

    return [field for field in QA_RAW_METADATA_FIELDS if field in fieldnames]


def rename_qa_metadata_fields(metadata: Dict[str, str]) -> Dict[str, str]:
    renamed = dict(metadata)

    if "source" in renamed:
        renamed["qa_source"] = renamed["source"]

    return renamed


def load_qa_documents(registry: Dict[str, Dict[str, str]]):
    documents = []

    for filename in sorted(os.listdir(QA_DIR)):
        if not filename.lower().endswith(".csv"):
            continue

        csv_path = os.path.join(QA_DIR, filename)
        csv_encoding = detect_text_encoding(csv_path)
        metadata_columns = get_csv_metadata_columns(csv_path)
        loader = CSVLoader(
            file_path=csv_path,
            metadata_columns=metadata_columns,
            encoding=csv_encoding,
        )

        registry_metadata = get_registry_metadata(filename, registry, default_doc_type="qa")
        qa_docs = loader.load()

        for doc in qa_docs:
            original_metadata = rename_qa_metadata_fields(doc.metadata)
            doc.metadata = clean_metadata({**registry_metadata, **original_metadata})

        documents.extend(qa_docs)

    return documents


def build_section_path(metadata: Dict[str, str]) -> str:
    sections = [metadata.get("h1"), metadata.get("h2"), metadata.get("h3")]
    cleaned_sections = [section.strip() for section in sections if section]
    return " > ".join(cleaned_sections)


def split_markdown_document(doc, chunk_size: int = 700, chunk_overlap: int = 100):
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=MARKDOWN_HEADERS_TO_SPLIT_ON)
    section_docs = header_splitter.split_text(doc.page_content)

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " "],
    )

    chunks = []
    for section_doc in section_docs:
        merged_metadata = clean_metadata({**doc.metadata, **section_doc.metadata})
        section_path = build_section_path(merged_metadata)
        if section_path:
            merged_metadata["section_path"] = section_path

        child_chunks = child_splitter.create_documents(
            texts=[section_doc.page_content],
            metadatas=[merged_metadata],
        )
        chunks.extend(child_chunks)

    return chunks


def load_markdown_documents(registry: Dict[str, Dict[str, str]]):
    documents = []

    for folder_name in SUPPORTED_MD_DIRS:
        folder_path = os.path.join(DATA_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for filename in sorted(os.listdir(folder_path)):
            if not filename.lower().endswith(".md"):
                continue

            file_path = os.path.join(folder_path, filename)
            file_encoding = detect_text_encoding(file_path)
            loader = TextLoader(file_path, encoding=file_encoding)
            md_docs = loader.load()
            registry_metadata = get_registry_metadata(
                filename,
                registry,
                default_doc_type=folder_name.rstrip("s"),
            )

            for doc in md_docs:
                doc.metadata = clean_metadata({**doc.metadata, **registry_metadata})
                documents.extend(split_markdown_document(doc))

    return documents


def load_all_documents():
    registry = load_metadata_registry()
    documents = []
    documents.extend(load_qa_documents(registry))
    documents.extend(load_markdown_documents(registry))

    for index, doc in enumerate(documents, start=1):
        doc.metadata = clean_metadata({**doc.metadata, "chunk_id": str(index)})

    return documents


def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR,
    )


def summarize_documents(documents):
    qa_count = sum(1 for doc in documents if doc.metadata.get("doc_type") == "qa")
    markdown_count = len(documents) - qa_count
    topics = sorted({doc.metadata.get("topic", "unknown") for doc in documents})
    qa_doc_types = sorted({doc.metadata.get("qa_doc_type", "unknown") for doc in documents if doc.metadata.get("doc_type") == "qa"})

    print("=" * 60)
    print("✅ 睡眠健康知识库构建完成！")
    print(f"🧩 最终文本块数：{len(documents)}")
    print(f"📚 QA文本块数：{qa_count}")
    print(f"📘 Markdown文本块数：{markdown_count}")
    print(f"🏷️ 已加载主题：{', '.join(topics)}")
    print(f"🗂️ QA类型：{', '.join(qa_doc_types)}")
    print("🗂️ 元数据字段：filename, title, doc_type, topic, source, qa_source, qa_doc_type, language, year, section_path(若有), chunk_id")
    print("=" * 60)


def main():
    documents = load_all_documents()
    build_vectorstore(documents)
    summarize_documents(documents)


if __name__ == "__main__":
    main()
