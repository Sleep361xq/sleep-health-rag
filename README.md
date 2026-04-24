# 睡眠健康智能问答系统

一个面向睡眠医学场景的智能助手原型，围绕睡眠健康问答、风险初筛、证据检索和个体数据分析结果解读构建。项目当前以垂直领域 RAG 为核心，并接入了睡眠分期与 OSA 分类工具，形成“知识问答 + 工具分析 + 结果解释”的产品雏形。

## 项目定位

这个项目不是通用聊天机器人，而是一个面向睡眠健康场景的垂直系统，重点解决以下几类问题：

- 睡眠医学知识问答
- 睡眠相关主诉的初筛与追问
- 指南、科普与结构化 QA 的可控检索
- 睡眠分期与 OSA 工具结果的页面化展示与解释

当前版本更适合定位为：

> 睡眠健康智能助手原型：RAG + 可控检索 + 初筛追问 + 个体睡眠数据分析工具接入

## 核心功能

### 1. 垂直领域 RAG 问答

- 基于睡眠医学指南、科普资料和结构化 QA 构建知识库
- 支持主题路由、查询改写、Self-query、重排序和证据展示
- 生成带引用、带边界说明的回答

### 2. 睡眠问题初筛与追问

- 对用户问题进行睡眠主题初筛
- 识别是否需要单轮追问
- 根据问题类型生成更有针对性的补充问题

### 3. 个体睡眠数据分析工具接入

- 支持上传 `.npz` 文件调用睡眠分期工具
- 支持上传 `.npz` 文件调用 OSA 分类工具
- 将工具结果展示为结构化摘要，并联动到后续问答解释中

### 4. 面向调试和展示的可解释界面

- 展示 screening 结果
- 展示 query rewrite 结果
- 展示 self-query 元数据约束
- 展示 rerank 候选信息
- 展示最终命中的证据与来源

## 系统能力概览

### RAG 主链路

```text
用户问题
-> 初筛判断（screening / follow-up）
-> 查询改写（query rewrite）
-> 主题路由（topic routing）
-> metadata 约束解析（self-query）
-> 向量检索（Chroma）
-> 重排序（CrossEncoder rerank）
-> LLM 生成回答
-> 证据与来源展示
```

### 工具分析链路

```text
上传 EEG/EOG 睡眠分期 .npz
-> 睡眠分期工具
-> 分期结果统计

上传 OSA 输入 .npz
-> OSA 分类工具
-> 风险分级与概率输出

工具结果摘要
-> 融合到问答页面
-> 用于结果解读与知识解释
```

## 技术栈

- Python
- Streamlit
- LangChain
- Chroma
- HuggingFace Embedding / Reranker
- DeepSeek API
- PyTorch
- 自定义睡眠分期与 OSA 推理工具

## 项目结构

```text
sleep_rag/
├─ app.py                         # Streamlit 前端
├─ run_app.py                     # 应用启动入口
├─ rag_router.py                  # RAG 检索与回答主链路
├─ screening.py                   # 初筛与 follow-up 判断
├─ query_rewriter.py              # 查询改写
├─ self_query.py                  # metadata 约束解析
├─ topic_router.py                # 主题路由
├─ reranker.py                    # 重排序
├─ analysis_tools.py              # 工具包装层
├─ data/                          # 指南、科普、QA、测试集与 metadata
├─ chroma_db/                     # 本地向量库
├─ outputs/evaluations/           # 自动评测输出
├─ tools/
│  ├─ eeg_sleep_staging_tool/     # EEG/EOG 睡眠分期工具
│  └─ osa_prediction_tool/        # OSA 分类工具
└─ docs/
   └─ architecture_overview.md    # 系统架构图与数据流说明
```

## 数据与知识来源

当前知识库包含三类核心数据：

- 指南类资料：AASM、国内失眠/OSA/CSA 相关指南与解读
- 科普类资料：中国睡眠研究会、国家卫健委等面向公众材料
- 结构化 QA：按主题整理的问答对，用于补强直接问答能力

核心元数据字段包括：

- `doc_type`
- `source`
- `topic`
- `year`

## 当前支持的产品场景

### 场景 1：纯知识问答

示例问题：

- AASM 关于 OSA 的指南怎么说？
- 长期睡眠不足会带来哪些危害？
- 失眠一般持续多久才算慢性？

### 场景 2：主诉初筛与追问

示例问题：

- 我最近总是半夜醒来很多次，是失眠吗？
- 我白天特别想睡，晚上还打呼噜，会不会有问题？

### 场景 3：个体分析结果解读

流程示例：

- 上传睡眠分期或 OSA 工具输入文件
- 运行工具分析
- 再提问：
  - 我的分期结果正常吗？
  - 为什么模型提示轻度 OSA？
  - 这种结果下一步建议做什么？

## 运行方式

### 1. 环境说明

当前项目联调时使用：

- `agent_py3_10`

### 2. 安装依赖

根据你的实际环境准备：

```bash
pip install -r requirements.txt
```

如果需要独立运行工具目录中的模型推理，也可以按工具目录的 `requirements.txt` 补依赖。

### 3. 配置项

主要配置位于：

- `config.py`

重点包括：

- LLM API Key / Base URL
- embedding 模型路径
- reranker 模型路径
- Streamlit 端口

### 4. 启动项目

```bash
E:\MyAnaconda\envs\agent_py3_10\python.exe run_app.py
```

启动后可在浏览器访问：

- `http://localhost:8501`

## 架构图

系统架构图和数据流说明见：

- [architecture_overview.md](docs/architecture_overview.md)

如果你的代码托管平台支持 Mermaid，也可以直接渲染文档中的图。

## 项目亮点

这个项目适合突出以下技术点：

- 垂直领域 RAG 设计与产品化落地
- rule + LLM 混合式主题路由与查询改写
- Self-query 元数据过滤与 rerank 融合检索
- 睡眠健康场景下的 follow-up 追问机制
- 工具增强型问答：睡眠分期与 OSA 分类结果联动解释
- 自动评测与定向回归驱动的迭代优化

## 当前边界

当前版本仍然是一个原型系统，边界包括：

- 工具结果用于辅助解释，不替代临床诊断
- 当前主要支持睡眠健康知识问答与单次分析展示
- 尚未形成完整的长期历史管理、自动报告体系和多用户系统
- 当前工具接入更偏“分析结果解释”，还不是完整的睡眠多智能体平台

## 后续可扩展方向

- 结果解读模板化
- 自动报告生成
- 长期睡眠历史管理
- 数据与 metadata 规范统一
- 多种信号输入的统一 schema
- 面向用户的趋势分析与风险提示



### 开源说明

本仓库公开版本主要用于展示系统架构、RAG 主链路设计、前端交互方式，以及工具增强型问答的工程接入思路。

以下内容已开源或建议开源：

- 睡眠健康 RAG 主链路与检索编排代码
- Streamlit 前端与交互逻辑
- screening / follow-up / query rewrite / self-query / rerank 模块
- 工具接入包装层与页面联动方式
- 系统架构文档、README 与演示说明

以下内容未开源或建议不在公开仓库中提供：

- 睡眠分期与 OSA 分类工具的核心课题实现
- 训练权重与训练代码
- 原始实验数据与受限样本
- 真实 API Key、本地模型绝对路径及其他敏感配置

如果你在公开仓库中看到 `tools/` 相关接口或占位说明，这些内容仅用于展示系统如何与外部算法工具集成，并不代表完整课题工具已经公开。

## 说明

本项目面向睡眠健康场景提供知识问答与分析解释支持，仅供研究与学习参考，不替代临床诊断与正式医疗意见。
