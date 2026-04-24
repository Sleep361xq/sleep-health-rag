import os
from pathlib import Path

# ==============================
# 基础目录配置
# ==============================
# 项目根目录
BASE_DIR = Path(__file__).resolve().parent
# 数据目录
DATA_DIR = BASE_DIR / "data"
# QA csv 所在目录
QA_DIR = DATA_DIR / "qa"
# 元数据注册表路径
REGISTRY_PATH = DATA_DIR / "metadata_registry.csv"
# Chroma 向量库持久化目录
VECTOR_DB_DIR = BASE_DIR / "chroma_db"

# ==============================
# 本地模型路径配置
# ==============================
# 向量化模型路径：用于构建向量库和检索时生成 embedding。
# 如果后续更换 embedding 模型，优先改这里或设置同名环境变量。
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", r"F:\LLM\emb_model\bge-small-zh-v1.5")

# 重排序模型路径：用于对初步召回结果进行 rerank。
# 如果后续更换 reranker 模型，改这里即可。
RERANK_MODEL = os.getenv("RERANK_MODEL", r"F:\LLM\emb_model\bge-reranker-base")

# ==============================
# 大语言模型（LLM）配置
# ==============================
# 使用的聊天模型名称
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
# 大模型服务的 base URL
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
# API Key：优先读取 LLM_API_KEY；如果没配，则兼容旧的 DEEPSEEK_API_KEY
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
# 生成温度：越低越稳定，越高越发散。当前项目建议保持低温。
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
# 单次回答最大 token 数
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# ==============================
# 检索与融合策略配置
# ==============================
# 文档型资料允许的 doc_type 列表
PDF_DOC_TYPES = ["guideline", "education", "other"]
# 最终结果里至少保留多少条 guideline 证据
FINAL_GUIDELINE_MIN = int(os.getenv("FINAL_GUIDELINE_MIN", "2"))
# 最终保留的 PDF / Markdown 证据上限
FINAL_PDF_LIMIT = int(os.getenv("FINAL_PDF_LIMIT", "6"))
# 最终保留的 QA 证据上限
FINAL_QA_LIMIT = int(os.getenv("FINAL_QA_LIMIT", "3"))
# 如果某轮 metadata filter 检索到的结果太少，则继续尝试 fallback 检索
MIN_FILTER_RESULTS = int(os.getenv("MIN_FILTER_RESULTS", "2"))

# ==============================
# Query Rewrite / 路由阈值配置
# ==============================
# 问题长度达到多少字符后，可能被视为复杂问题
COMPLEX_LENGTH_THRESHOLD = int(os.getenv("COMPLEX_LENGTH_THRESHOLD", "28"))
# 命中多少个复杂信号后，判为复杂问题
COMPLEXITY_SIGNAL_THRESHOLD = int(os.getenv("COMPLEXITY_SIGNAL_THRESHOLD", "2"))
# 最多允许拆分出多少个子查询
MAX_SUB_QUERIES = int(os.getenv("MAX_SUB_QUERIES", "4"))
# topic router 里规则命中的最小置信阈值
TOPIC_RULE_CONFIDENT_THRESHOLD = int(os.getenv("TOPIC_RULE_CONFIDENT_THRESHOLD", "1"))

# ==============================
# 构库相关静态配置
# ==============================
# 支持加载的 Markdown 子目录
SUPPORTED_MD_DIRS = ["guidelines", "education", "others"]
# QA csv 中会保留为 metadata 的原始字段
QA_RAW_METADATA_FIELDS = ["topic", "source"]
# 自动检测文本编码时尝试的编码列表
TEXT_ENCODINGS = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
# Markdown 切分时识别的标题层级
MARKDOWN_HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]

# ==============================
# Web 服务启动配置
# ==============================
# 监听地址：
# - 127.0.0.1 表示只能本机访问
# - 0.0.0.0 表示局域网内其他设备也可访问
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
# Streamlit 服务端口
APP_PORT = int(os.getenv("APP_PORT", "8501"))
# 是否以 headless 模式启动（一般保持 true 即可）
APP_HEADLESS = os.getenv("APP_HEADLESS", "true").lower() == "true"
