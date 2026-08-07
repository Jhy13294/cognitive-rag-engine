# Cognitive RAG Engine

[![Offline Test Gate](https://github.com/Jhy13294/cognitive-rag-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/Jhy13294/cognitive-rag-engine/actions/workflows/tests.yml)

## 一句话

一个面向企业知识库场景、工程化的 RAG 系统：检索漏斗与生成质量都由评估门禁把关。重点在可评估检索、混合检索、重排、ACL 检索前过滤、缓存、可观测性和异步 FastAPI 服务——不是一个单文件 demo。

## 为什么做这个项目

大多数 RAG 示例止步于「PDF → embedding → 相似度检索 → 回答」。这足够跑一个 notebook，却把决定检索是否正确、答案是否有据、系统是否能安全上线的每一个决策都藏了起来。本项目把这些当成真正要做的工作：检索质量每次 push 都对着冻结基线度量，生成质量由 LLM-as-judge 轨评分，权限过滤前置于候选评分，服务自带缓存、可观测性和健康探针。它是 production-shaped（生产形态），而非 production-complete（生产完备）——边界在 [生产化边界](#生产化边界) 里写清楚。

## 系统架构

```text
文档入库（TXT / Markdown / PDF+OCR / Word）
  → 清洗 / 切分（父子分块）
  → embedding
  → dense 检索 + BM25 稀疏检索
  → RRF 混合融合
  → 重排
  → 父块展开
  → 上下文装填
  → 带引用生成
  → 评估 / 指标 / 缓存 / ACL 检索前过滤
```

ACL 检索前过滤、Redis 缓存和可观测性是包裹这条漏斗、而非嵌在其中：权限过滤前置于评分，缓存包裹 retrieve/answer 调用，审计/指标是 fail-open 旁路。数据流细节见 [docs/architecture.md](docs/architecture.md)。

## 核心亮点

- **可评估检索** —— 确定性 golden set 评估（hit rate、MRR、recall、负样本指标），逐位可复现的冻结基线，CI 门禁每次 push 对 MRR@3 做六位小数精确断言。
- **混合检索** —— Dense + BM25 经 RRF 融合，外加确定性多查询改写与跨查询融合。
- **重排与父子分块** —— 可插拔重排、失败降级回 dense；子块用于检索，父块展开供给生成。
- **上下文装填** —— 生成输入侧的整块纳入、确定性去重、引用连号和 token 预算。
- **异步 FastAPI 服务** —— `/ingest`、`/query` 与 SSE `/query/stream`，同步漏斗走线程池 offload、异常统一脱敏映射。
- **Redis 缓存与 ACL/RBAC 检索前过滤** —— 三层缓存（query embedding / 检索结果 / 答案）+ 原子 corpus version 失效；ACL fail-closed 且前置于候选评分，有效 ACL 进入缓存 key。
- **可观测性与健康门禁** —— 有界 label 的 Prometheus 指标、脱敏 JSON-line 审计、request ID、liveness/readiness 双探针；CI 把测试账目钉死以防「静默跳过伪装全绿」。

## 快速开始

最快一瞥，无需 Docker、无需 API key —— 在 golden set 上跑离线检索评估：

```bash
pip install -r requirements.txt
python -m eval.run
```

想看完整端到端流程（离线评估 **加** 一个零密钥全栈 HTTP demo，含 ACL、引用、缓存计数和健康探针），见 **[examples/minimal_demo.md](examples/minimal_demo.md)**。

## 评估

检索质量与生成质量分两轨评估，离线回放是默认的可复现路径：

- **确定性检索评估** —— `python -m eval.run` 用 `HashEmbeddingProvider` 在 5 篇 765–887 字节的合成 Markdown 上评估 20 条标注 query（16 条正样本、4 条负样本）。CI 每次实跑 dense、hybrid、parent-child、rerank 四个 profile，对正样本 MRR@3 做六位小数精确断言；负样本行为单独报告。
- **生成质量（Ragas）** —— live LLM-as-judge 轨对 Faithfulness、Answer Relevance、Context Precision/Recall 打分，温度钉死、多次重复、median 门禁；CI 只离线回放冻结 verdict fixture。分数是小样本上带方差的判官估计，不是 production 质量下限。

详见下方 [生成质量评估](#生成质量评估)，零付费本机延迟/QPS 口径见 [docs/benchmark.md](docs/benchmark.md)。

## 生产化边界

这是一个面向企业、生产形态（production-shaped）的 RAG 后端，而非开箱即用的 production-ready 产品。边界是刻意的，且事先讲清：

- **内置认证尚未实现。** 服务不校验 JWT 或 session。
- **ACL 依赖受信上游网关。** principal 从受信上游 header 读取；必须部署在一个持有该 header 的认证网关之后。body 传入的 principal 默认拒绝，除非为本地测试显式开启。
- **离线回放/golden 基线是默认可复现路径。** live LLM-as-judge 评估和付费检索需要 API key；在 CI 里无密钥跑的是冻结基线和离线门禁。
- **评估分数是小样本上的判官估计。** golden set 约 20 条人工标注查询；分数当方向参考、不是 production 质量保证。

## 当前进度

当前阶段：可评估的检索漏斗、异步 HTTP 服务层、可选 Redis 缓存、ACL/RBAC 检索前过滤、fail-open 可观测性，以及部署健康门禁。

当前能力高光：

- CI 对 dense、hybrid、parent-child、rerank 四个确定性 profile 的 MRR@3 做六位小数精确断言。
- TXT、Markdown、PDF/OCR、Word 入库统一生成原文连续切片；Markdown fenced code 默认进入可检索正文。
- Dense、BM25、RRF、重排、Multi-Query、父块展开和 token-aware 上下文装填共用一条分阶段管线。
- 异步 FastAPI 服务提供入库、问答、SSE 流式响应、Redis 缓存、ACL/RBAC 检索前过滤、可观测性和双健康探针。
- Memory provider 支持无密钥本地验证；Qdrant、MySQL、Redis、OpenAI-compatible 生成、OpenAI embedding 与 Cohere 重排均为可选集成。

完整清单与明确缺口见[能力清单](docs/capabilities.zh-CN.md)。内置认证、JWT 校验和 session 管理仍不在当前范围内。

## 高级学习笔记

想理解项目背后的工程取舍时，可以从这里开始：

- **[技术选型思考](docs/tech-selection.md)**：为什么选择当前 embedding provider、向量库、检索栈、服务协议、访问过滤、缓存和可观测性设计。
- **[系统架构](docs/architecture.md)**：系统架构、ingest/query 数据流、检索漏斗、访问控制、缓存布局、可观测性和评估边界。
- **[冻结检索基线](docs/retrieval-baselines.md)**：当前 CI 数值、Parent-Child 迁移差异、同分排序约束与依赖锁口径。
- **[核心踩坑记录](docs/dev-log-crashing.md)**：关于指标污染、假流式、异步陷阱、缓存失效、访问边界、审计泄漏和 token 账目等问题的调试记录。
- **[本机基准](docs/benchmark.md)**：面向零密钥 compose 栈的可复跑延迟/QPS 口径；数字不包含真实付费生成。

## 目录结构

```text
.
├── api_client.py              # LLM API 客户端
├── config.py                  # 环境变量配置
├── logger.py                  # 日志工具
├── main.py                    # 命令行问答入口
├── rag_cli.py                 # RAG 应用命令行入口
├── service/                   # FastAPI HTTP 适配层
├── cache/                     # Redis 缓存装饰器和序列化
├── access/                    # ACL/RBAC filter、resolver 和 MySQL metadata 适配器
├── observability/             # 结构化审计、指标 registry、request ID 和告警钩子
├── ci/                        # push CI 门禁：全量套件账目与冻结检索基线
├── .github/workflows/         # CI：离线测试门禁、Ragas replay 门禁、gated live 评估
├── Dockerfile                 # 服务镜像构建
├── docker-compose.yml         # 本地全栈：Qdrant、Redis、MySQL 和应用
├── LICENSE                    # MIT License
├── requirements.txt           # 开发依赖
├── requirements-ci.in        # CI 精确直接依赖
├── requirements-ci.constraints # 跨平台 transitive 兼容约束
├── requirements-ci.txt       # universal CI 完整锁
├── requirements-serve.txt     # 服务镜像锁定的运行时依赖
├── document_loader/
│   ├── base.py                # Document 模型和加载器接口
│   ├── chunking.py            # 文本切分
│   ├── loader.py              # 统一加载入口
│   ├── md_loader.py           # Markdown 加载器
│   ├── pdf_loader.py          # 支持扫描页检测、OCR 兜底和表格的 PDF 加载器
│   ├── txt_loader.py          # TXT 加载器
│   └── word_loader.py         # Word 加载器
├── embeddings/
│   ├── base.py                # Embedding 接口和向量工具
│   ├── openai_provider.py     # OpenAI Embedding Provider
│   └── hash_provider.py       # 用于测试的本地确定性 Provider
├── vector_store/
│   ├── base.py                # 向量库接口和记录模型
│   ├── factory.py             # 向量库工厂
│   ├── memory_store.py        # 内存向量库
│   └── qdrant_store.py        # Qdrant 向量库适配器
├── parent_store/
│   ├── base.py                # 父块键值存储接口
│   └── memory_store.py        # 内存父块存储
├── tokenization/
│   └── counter.py             # TokenCounter、tiktoken 计数器和离线兜底
├── eval/
│   ├── golden_set.jsonl       # 使用 relevant 列表标注的检索 golden set
│   ├── baseline.py            # HashEmbeddingProvider 确定性基线
│   ├── metrics.py             # hit_rate、MRR、recall、negative 指标
│   ├── reporting.py           # JSON 和 Markdown 报告
│   ├── run.py                 # python -m eval.run 入口
│   ├── ragas_run.py           # python -m eval.ragas_run replay/live/compare 入口
│   ├── ragas_evaluation.py    # Ragas replay 门禁、fixture 校验与门禁规则
│   ├── ragas_live.py          # gated live 判官录制
│   ├── bge_embedding.py       # answer relevance 本地 fastembed BGE embedding
│   ├── gemini_embedding.py    # answer relevance 可选 Gemini embedding
│   ├── fixtures/              # 评估知识库与冻结 fixture
│   │   ├── ragas_verdicts.jsonl   # 已提交的 Ragas 判官冻结基线（replay 门禁输入）
│   │   └── query_rewrites.jsonl   # 确定性 query rewrite fixture
│   └── reports/               # 生成的评估报告
├── bench/
│   ├── run.py                 # python -m bench.run 零付费本机延迟/QPS 基准入口
│   └── reports/               # 生成的基准报告（git 忽略）
├── examples/
│   ├── minimal_demo.md        # 离线评估 + 零密钥全栈 HTTP demo
│   ├── demo_chat_stub.py      # demo 用的本地确定性聊天 stub
│   └── docker-compose.demo.yml # demo 用的零密钥 compose override
├── lexical/
│   ├── tokenizer.py           # 共享 normalization 和 tokenization
│   └── bm25.py                # IDF、BM25 和词法评分 primitives
├── hybrid/
│   ├── models.py              # 排序检索结果模型
│   ├── bm25_retriever.py      # 基于 VectorRecord corpus 的 BM25 检索器
│   └── rrf.py                 # Reciprocal Rank Fusion
├── rerank/
│   ├── base.py                # Reranker 接口和结果模型
│   ├── deterministic.py       # 离线确定性词法重排器
│   ├── cohere_provider.py     # 带重试的 Cohere Rerank Provider
│   └── factory.py             # Reranker 工厂
├── query_rewrite/
│   ├── base.py                # Query rewrite 接口和配置
│   ├── deterministic.py       # 离线 fixture 改写器
│   ├── chat.py                # Chat 改写器
│   └── factory.py             # Query rewriter 工厂
├── rag/
│   ├── pipeline.py            # 高层 RAG 编排与兼容导出
│   ├── retrieval_orchestrator.py # dense/hybrid/multi-query 检索与重排
│   ├── source_finalizer.py    # 父块展开与兄弟子块折叠
│   ├── prompt_builder.py      # 系统提示、scope 指令、上下文组装
│   ├── question_classifier.py # 确定性问题形状启发式
│   ├── models.py              # 共享 source/response 数据模型
│   └── context_packing.py     # top-k 后的上下文装填
├── text_cleaner/
│   └── cleaner.py             # 文本清洗
├── tests/
│   ├── fixtures/              # TXT 和 Markdown 样例文档
│   ├── test_acl.py
│   ├── test_acl_mysql_integration.py
│   ├── test_api_client.py
│   ├── test_bge_embedding.py
│   ├── test_cache.py
│   ├── test_ci_gates.py
│   ├── test_context_packing.py
│   ├── test_document_ingestion.py
│   ├── test_e2e_integration.py
│   ├── test_embeddings.py
│   ├── test_eval_metrics.py
│   ├── test_gemini_embedding.py
│   ├── test_hybrid.py
│   ├── test_observability.py
│   ├── test_parent_child.py
│   ├── test_qdrant_store_integration.py
│   ├── test_qdrant_store_mock.py
│   ├── test_query_rewrite.py
│   ├── test_rag_cli.py
│   ├── test_rag_pipeline.py
│   ├── test_ragas_eval.py
│   ├── test_rerank.py
│   ├── test_service.py
│   ├── test_token_counter.py
│   └── test_vector_store.py
└── docs/
    ├── capabilities.md        # 英文能力与缺口清单
    ├── capabilities.zh-CN.md  # 中文能力与缺口清单
    ├── tech-selection.md      # 技术选型思考
    ├── architecture.md        # 系统架构与数据流图
    ├── retrieval-baselines.md # 冻结指标与可复现约束
    ├── dev-log-crashing.md    # 核心踩坑记录
    └── learning_notes.zh-CN.md # 学习笔记索引
```

## 安装

创建并激活虚拟环境后，安装依赖：

```bash
pip install -r requirements.txt
```

复现 CI 依赖集时安装 `requirements-ci.txt`。维护者在 `requirements-ci.in` 更新直接依赖，在 `requirements-ci.constraints` 更新兼容约束，并按 lock 文件头命令重新生成 universal lock。

创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions

EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL_NAME=text-embedding-3-small
EMBEDDING_DIMENSION=512

PDF_EXTRACT_TABLES=false
PDF_OCR_ENABLED=false
PDF_OCR_MIN_CHARS=1
PDF_OCR_DPI=200

VECTOR_STORE_PROVIDER=memory
VECTOR_STORE_COLLECTION=enterprise_kb

RERANK_ENABLED=false
RERANK_PROVIDER=deterministic
RERANK_FETCH_K=30
RERANK_TOP_N=5

HYBRID_ENABLED=false
HYBRID_DENSE_WEIGHT=0.2
HYBRID_SPARSE_WEIGHT=1.0
RRF_K=60
BM25_K1=1.5
BM25_B=0.75

PARENT_CHILD_ENABLED=false
PARENT_CHUNK_SIZE=1600
PARENT_CHUNK_OVERLAP=200
CHILD_CHUNK_SIZE=400
CHILD_CHUNK_OVERLAP=80

CONTEXT_PACKING_ENABLED=false
CONTEXT_DEDUP_ENABLED=false
CONTEXT_NEAR_DUP_ENABLED=false
CONTEXT_NEAR_DUP_THRESHOLD=0.9
CONTEXT_MAX_TOKENS=2048
TOKENIZER_ENCODING=cl100k_base

QUERY_REWRITE_ENABLED=false
QUERY_REWRITE_PROVIDER=deterministic
QUERY_REWRITE_FIXTURE_PATH=eval/fixtures/query_rewrites.jsonl
QUERY_REWRITE_NUM_QUERIES=3
QUERY_REWRITE_TEMPERATURE=0.1
QUERY_REWRITE_CACHE_ENABLED=true
QUERY_REWRITE_WEIGHT_ORIGINAL=1.0
QUERY_REWRITE_WEIGHT_VARIANT=0.7

REDIS_URL=redis://localhost:6379/0
CACHE_NAMESPACE=rag-cache
CACHE_ENABLED=false
CACHE_EMBEDDING_ENABLED=true
CACHE_RETRIEVAL_ENABLED=true
CACHE_ANSWER_ENABLED=true
CACHE_EMBEDDING_TTL=604800
CACHE_RETRIEVAL_TTL=900
CACHE_ANSWER_TTL=300
CACHE_TIMEOUT=0.25

ACL_ENABLED=false
ACL_METADATA_KEY=acl
ACL_DEFAULT_DENY=true
ACL_PRINCIPAL_HEADER=X-Principal
ACL_ALLOW_BODY_PRINCIPAL=false
ACL_INGEST_BINDINGS_ENABLED=false
METADATA_DB_URL=mysql://user:password@localhost:3306/rag_metadata

OBSERVABILITY_ENABLED=false
AUDIT_ENABLED=false
METRICS_ENABLED=false
AUDIT_LOG_PATH=logs/audit.jsonl
AUDIT_LOG_QUERY_TEXT=false
AUDIT_PRINCIPAL_MODE=hash
AUDIT_HASH_SALT=replace-with-a-long-random-secret
METRICS_NAMESPACE=rag
METRICS_PATH=/metrics
READINESS_TIMEOUT=2.0
```

## 使用

运行基础命令行问答：

```bash
python main.py
```

使用 RAG CLI 进行一次性提问：

```bash
python rag_cli.py tests/fixtures --question "What is this project?"
```

进入 RAG CLI 交互模式：

```bash
python rag_cli.py tests/fixtures
```

先构建索引，再查询已有向量库 collection：

```bash
python rag_cli.py ingest knowledge_base \
  --embedding-provider openai \
  --embedding-dimension 512 \
  --vector-store qdrant

python rag_cli.py query "What does the knowledge base say about deployment?" \
  --embedding-provider openai \
  --embedding-dimension 512 \
  --vector-store qdrant
```

`memory` 向量库是进程内状态，只适合职责分离测试；跨进程持久化请使用 Qdrant。

**索引迁移（2026 年 8 月）：** 精确原文 span、Markdown 代码可检索语义与 source 路径分隔符归一化会同时改变 chunk 正文或 record id。已有持久化 Qdrant collection 必须重建一次后再入库；仅在该次 ingest 设置 `VECTOR_STORE_RECREATE=true`，完成后恢复为 `false`。普通 upsert 不会删除旧 id 对应的记录。

本地启动 HTTP 服务：

```bash
uvicorn service.app:app --host 127.0.0.1 --port 8000
```

服务提供 `POST /ingest`、`POST /query`、`POST /query/stream` SSE 流式接口，以及 `GET /cache/stats`、`GET /health` 和 `GET /ready`；启用指标后还会提供 `GET /metrics`。当前尚未内置鉴权，不能在没有受信网关的情况下直接暴露到公网。

Redis 缓存默认关闭。启用时需要运行 Redis，设置 `CACHE_ENABLED=true` 并配置 `REDIS_URL`。`/query/stream` 在 L3 命中时会重放缓存答案，并在 SSE payload 中标记 `cached=true`；未命中时保持真实 provider 流式输出，并在完成后回填 L3。

ACL/RBAC 过滤默认关闭。启用时设置 `ACL_ENABLED=true`，配置 metadata 数据库，并把服务放在认证网关后，由网关写入 `ACL_PRINCIPAL_HEADER` 指定的受信 principal header（默认 `X-Principal`）。请求体中的 principal 默认拒绝，只有本地测试显式设置 `ACL_ALLOW_BODY_PRINCIPAL=true` 时才允许 fallback。设置 `ACL_INGEST_BINDINGS_ENABLED=true` 后，ingest 可把 MySQL ACL binding 写入向量 payload；Qdrant payload 仍是检索执行快照，因此 MySQL binding 变更需要 re-ingest 或 re-sync 后才会影响检索。

可观测性默认关闭。启用时需要设置总闸 `OBSERVABILITY_ENABLED=true`，并至少开启 `AUDIT_ENABLED` 或 `METRICS_ENABLED`；审计 hash 还必须提供部署侧秘密 `AUDIT_HASH_SALT`。运行时 audit/metric 故障 fail-open，但开启后的非法配置会在应用创建时早失败。query 默认只落 hash，只有显式设置 `AUDIT_LOG_QUERY_TEXT=true` 才记录定长原文。token 指标优先使用 provider reported 计数；上游没有 usage 时明确标为 `estimated`，provider watermark 会防止并发请求或缓存 raw response 重复累计 token。embedding 估算只覆盖服务边界可见的 query 文本，provider 侧 rewrite 扩展仍是估算，不能当作计费真账。

使用 Docker Compose 启动本地全栈：

```bash
cp .env.compose.example .env
# 启动前填写 DEEPSEEK_API_KEY、EMBEDDING_API_KEY 和 AUDIT_HASH_SALT。
docker compose up --build
docker compose ps
```

该 compose 栈会启动 Qdrant、Redis、MySQL 和 FastAPI app，容器间使用服务名互联（`qdrant`、`redis`、`mysql`）。app 容器启动时会执行幂等初始化：创建 MySQL ACL schema、写入 demo principal 与 ACL binding，并用受信 `--acl` 元数据把 `eval/fixtures/knowledge_base/*.md` 写入 Qdrant。这是全栈部署 smoke 路径，不是零密钥问答；ingest 需要 embedding API key，`/query` 需要有效 DeepSeek key。

四个 healthcheck 都变绿后，可以尝试：

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/ready

curl -s http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-Principal: alice" \
  -d '{"question":"What finance approval rules are in the knowledge base?","top_k":3}'

curl -s http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-Principal: bob" \
  -d '{"question":"What finance approval rules are in the knowledge base?","top_k":3}'

curl -s http://localhost:8000/metrics
```

`alice` 默认拥有 finance 权限，`bob` 默认拥有 legal 权限。重复执行 `docker compose up` 时，初始化逻辑应再次完成且不重复写 ACL 行或向量记录。`/health` 用作 liveness 探针，`/ready` 用作 readiness 探针：Kubernetes 的 `livenessProbe` 应指向 `/health`，避免后端短暂抖动时重启进程；`readinessProbe` 应指向 `/ready`，让 Qdrant、Redis 或 MySQL 断连时以 HTTP 503 fail-closed 停止接流量。

常用 RAG CLI 参数：

```bash
python rag_cli.py knowledge_base \
  --question "What does the knowledge base say about deployment?" \
  --embedding-provider openai \
  --embedding-dimension 512 \
  --vector-store qdrant \
  --top-k 5 \
  --hybrid \
  --hybrid-fetch-k 30 \
  --hybrid-dense-weight 0.2 \
  --hybrid-sparse-weight 1.0 \
  --parent-child \
  --parent-chunk-size 1600 \
  --child-chunk-size 400 \
  --context-packing \
  --context-dedup \
  --context-max-tokens 2048 \
  --multi-query \
  --query-rewrite-provider deterministic \
  --rerank-provider deterministic \
  --rerank-fetch-k 30 \
  --chunk-size 800 \
  --chunk-overlap 120 \
  --principal alice \
  --metadata-filter "{\"file_type\":\"markdown\"}"
```

通过统一入口加载文档：

```python
from document_loader import load_document

document = load_document("example.md", clean=True)
```

加载并切分文档：

```python
from document_loader import load_and_split_document

chunks = load_and_split_document(
    "example.md",
    clean=True,
    chunk_size=800,
    chunk_overlap=120,
)
```

加载目录下所有支持的文档：

```python
from document_loader import load_documents

documents = load_documents("knowledge_base", recursive=True, clean=True)
```

使用 OpenAI Embedding Provider 生成向量：

```python
from document_loader import load_and_split_document
from embeddings import OpenAIEmbeddingProvider

chunks = load_and_split_document("example.md", clean=True)
provider = OpenAIEmbeddingProvider(dimensions=512)
embedded_chunks = provider.embed_documents(chunks)
```

将向量写入内存向量库并检索：

```python
from document_loader import load_and_split_document
from embeddings import OpenAIEmbeddingProvider
from vector_store import InMemoryVectorStore

chunks = load_and_split_document("example.md", clean=True)
provider = OpenAIEmbeddingProvider(dimensions=512)
embedded_chunks = provider.embed_documents(chunks)

store = InMemoryVectorStore(dimension=provider.dimension)
store.add_documents(embedded_chunks)

query_embedding = provider.embed_text("What does this document say about RAG?")
results = store.similarity_search(query_embedding, top_k=3)
```

使用 Qdrant 向量库：

```python
from document_loader import load_and_split_document
from embeddings import OpenAIEmbeddingProvider
from vector_store import QdrantVectorStore

chunks = load_and_split_document("example.md", clean=True)
provider = OpenAIEmbeddingProvider(dimensions=512)
embedded_chunks = provider.embed_documents(chunks)

store = QdrantVectorStore(
    collection_name="enterprise_kb",
    dimension=provider.dimension,
    url="http://localhost:6333",
)
store.add_documents(embedded_chunks)

query_embedding = provider.embed_text("What does this document say about RAG?")
results = store.similarity_search(query_embedding, top_k=3, metadata_filter={"file_type": "markdown"})
```

运行最小 RAG 管线：

```python
from api_client import APIClient
from config import Config
from document_loader import load_and_split_document
from embeddings import OpenAIEmbeddingProvider
from rag import RAGPipeline
from vector_store import InMemoryVectorStore

chunks = load_and_split_document("example.md", clean=True)
provider = OpenAIEmbeddingProvider(dimensions=512)
embedded_chunks = provider.embed_documents(chunks)

store = InMemoryVectorStore(dimension=provider.dimension)
store.add_documents(embedded_chunks)

client = APIClient(Config.API_KEY, Config.API_URL)
pipeline = RAGPipeline(provider, store, client)

response = pipeline.answer("What does this document say about RAG?")
print(response.answer)
print(response.sources)
```

使用确定性离线重排器运行管线：

```python
from rerank import DeterministicReranker

reranker = DeterministicReranker(top_n=5, fetch_k=30)
pipeline = RAGPipeline(provider, store, client, top_k=5, reranker=reranker, fetch_k=30)
```

使用 hybrid dense + BM25 检索运行管线：

```python
from hybrid import BM25Retriever, RRFConfig, ReciprocalRankFusion

bm25 = BM25Retriever(vector_records)
rrf = ReciprocalRankFusion(
    RRFConfig(k=60, weights={"dense": 0.2, "sparse": 1.0})
)
pipeline = RAGPipeline(
    provider,
    store,
    client,
    top_k=5,
    fetch_k=30,
    bm25_retriever=bm25,
    rrf=rrf,
)
```

启用上下文装填：

```python
pipeline = RAGPipeline(
    provider,
    store,
    client,
    top_k=5,
    context_packing_enabled=True,
    context_dedup_enabled=True,
    context_max_tokens=2048,
    tokenizer_encoding="cl100k_base",
)
```

使用 Parent-Child 父块展开运行管线：

```python
from document_loader import load_and_split_documents_hierarchical
from embeddings import OpenAIEmbeddingProvider
from parent_store import InMemoryParentStore
from rag import RAGPipeline
from vector_store import InMemoryVectorStore

split = load_and_split_documents_hierarchical(
    "knowledge_base",
    parent_chunk_size=1600,
    parent_chunk_overlap=200,
    child_chunk_size=400,
    child_chunk_overlap=80,
)

parent_store = InMemoryParentStore()
parent_store.add_parents(split.parents)

provider = OpenAIEmbeddingProvider(dimensions=512)
embedded_children = provider.embed_documents(split.children)

store = InMemoryVectorStore(dimension=provider.dimension)
store.add_documents(embedded_children)

pipeline = RAGPipeline(
    provider,
    store,
    client,
    parent_store=parent_store,
)
```

运行测试：

```bash
python -m unittest discover -s tests -v
```

运行确定性检索基线：

```bash
python -m eval.run
```

运行确定性重排评估：

```bash
python -m eval.run --rerank-provider deterministic --rerank-fetch-k 30 --rerank-top-n 10
```

在关闭 reranker 的前提下运行 hybrid 三路对照：

```bash
python -m eval.run --compare-hybrid --hybrid-fetch-k 30
```

运行 Parent-Child 子块语料检索评估：

```bash
python -m eval.run --parent-child
```

运行确定性 Multi-Query 检索评估：

```bash
python -m eval.run --multi-query
```

强制评估绕过 Redis 缓存装饰器：

```bash
python -m eval.run --no-cache
```

评估器只依赖 `retrieve(question, top_k)` 可调用对象，不调用聊天模型。默认基线使用 `HashEmbeddingProvider`，因此可以离线复现。

冻结 fixture 包含 5 篇 765–887 字节的合成 Markdown，以及 20 条 query（16 条正样本、4 条负样本）。下述 MRR 与 recall 只针对正样本。top-k 为 3 时，4 条负样本仍全部返回非空结果（`negative_false_recall_rate = 1.000000`），因此当前系统尚未证明弃答能力。

当前重排基线使用 `HashEmbeddingProvider` + `DeterministicReranker`：整体 MRR@3 从 `0.677083` 提升到 `1.000000`，long_tail MRR@3 从 `0.566667` 提升到 `1.000000`，recall@5 保持 `1.000000`。`ci/retrieval_baseline_gate.py` 会断言 rerank profile；报告在每次本地运行时生成，并由 CI 作为 workflow artifact 上传。

当前 hybrid retrieval 对照在关闭 reranker、使用 `HashEmbeddingProvider` 的条件下：dense-only MRR@3 为 `0.677083`，bm25-only MRR@3 为 `1.000000`，fused MRR@3 为 `1.000000`；fused 的 exact_name 和 long_tail MRR@3 也达到 `1.000000`，并保持指标输出确定。hybrid profile 由同一门禁断言，报告按次生成而不提交入库。

当前 Parent-Child retrieval 模式只让子块进入检索和排序，父块只在最终 top-k 后展开供生成使用。评估命令默认关闭父块展开，只评子块 ranked list；因此检索指标的浮动属于“子块粒度变化”，不能记为父块展开带来的提升。精确原文切片语义下冻结 MRR@3 为 `0.614583`；旧值 `0.625000` 依赖 span 错误时的 fallback 切片，现已退役。生成侧连贯性和 Context Precision 的真账留到 Ragas 质量评估。

当前上下文装填只属于生成输入组装，不进入 `retrieve()`，不能被拿来声明 MRR 或 recall 提升。所有 context flag 关闭时，旧的字符制 `_build_context` 路径保持兼容；显式开启后，ContextPacker 会整块纳入或跳过，引用重新连号，精确重复默认由开关控制，可选近重复去重由独立 flag 守卫，并使用 TokenCounter 计算 token 预算。tiktoken 是可选依赖，缺失时自动使用确定性启发式兜底。对中文文本，真实 token 计数可能让 batch 变多、变小但合法；收益是避免超限请求，而不是“批数下降”。生成质量和连贯性结论仍延后到 Ragas 质量评估。

当前 Multi-Query retrieval 是检索侧改动，因此 hit_rate、MRR、recall 是合法测量面。确定性 fixture 保证原始 query 始终作为 `q0`，只给 paraphrase 和 long_tail 样本增加冻结改写变体，多路检索后复用既有 RRF 融合，再进入既有 rerank、父块展开和上下文装填。离线 HashEmbeddingProvider 基线下，`python -m eval.run --multi-query` 将 paraphrase recall@3 从 `0.800000` 提升到 `1.000000`，long_tail recall@3 保持 `0.900000`，long_tail MRR@3 从 `0.566667` 提升到 `0.900000`。4 条 negative query 没有 fixture 改写，因此 multi-query 对它们整段旁路，结果与单路基线逐字节一致；multi-query 在 negative 上触发的风险没有被离线门禁覆盖，后续交给 roadmap 中的阈值 / abstain 工作处理。这些确定性 fixture 涨幅只是“给定已知优质改写时，RRF 融合管线能带来目标 paraphrase/long_tail 收益”的机制受控演示。线上 LLM 改写可能高于也可能低于这组数字，query drift 甚至可能跌破单路；这不是生产保底。

当前 Redis 缓存是包在现有漏斗外的装饰层，不是第二条检索管线。`CACHE_ENABLED=false` 时 provider 和 pipeline 不会被包装；启用后，L1 包 `EmbeddingProvider.embed_text`，L2 包 `retrieve`，L3 包 `answer`，`rag/pipeline.py` 不改。L1 不含 `corpus_version`，因为文本向量是 model+text 的函数；L2/L3 必须含 Redis 持久化 corpus version，使 `/ingest` 后旧检索和旧答案跨进程、跨副本都不可达。negative/abstain 类结果按精确 query 正常缓存，但受 TTL 和 corpus version 双重约束。默认单测使用 FakeRedis 替身；真实 `redis.asyncio` 覆盖必须由 `REDIS_URL` 守卫，并同时覆盖真实 Redis 命令和跨事件循环调用。
真实 Redis 路径已通过生产形态回归：连续同步调用和服务线程池调用会复用 store 自有 Redis loop，第二次命中 L1/L2/L3，并保持 Redis error 计数为 0。

当前 ACL/RBAC 支持的是 fail-closed 检索前过滤，不是检索后过滤。有效 ACL filter 由服务端根据受信 principal 构造，再与客户端 metadata filter 做 AND，因此客户端只能收窄结果，不能放宽权限。记录授权条件为 `record.acl` 与解析出的 allowed ACL subjects 有交集；缺失或空 ACL 的记录对普通用户视为受限。同一个 filter 会在 dense scoring、BM25 sparse scoring、multi-query 每个变体和缓存 key 构造前生效。MySQL 作为 principal membership 和可选 document/chunk ACL binding 的 metadata 真相源，Qdrant payload 作为高性能检索执行快照；修改 MySQL binding 后需要 re-ingest 或 re-sync 才会进入向量库 payload。FastAPI 服务本身不验证 JWT 或 session，生产部署必须由受信认证网关持有 principal header。

当前可观测性是服务接缝外层能力，不参与检索或授权决策。结构化审计只记录生成的 request ID、opaque principal/query 标识、有限请求元数据和稳定 record ID，不记录 source 正文或 ACL subjects。Prometheus label 仅限 route、outcome、retrieval mode、token source 和 cache layer；request ID、principal、query 原文和 source ID 都不得成为 label。空召回、ACL 拒绝和服务错误提供稳定 alert code 与可插拔 hook。所有观测故障都 fail-open，这与 ACL 安全边界的 fail-closed 是刻意相反的语义。

## 生成质量评估

生成质量采用 Ragas 四维：Faithfulness、Answer Relevance、Context Precision、Context Recall。它与确定性检索评估严格隔离：`python -m eval.run` 继续保持离线、逐字节可复现；Ragas 只把 JSON 报告写入 `eval/reports/ragas/`。

live 判官复用项目的 DeepSeek OpenAI-compatible API 配置；Answer Relevance 的 embedding 走可选 provider（`RAGAS_EMBEDDING_PROVIDER`）。`bge` 走本地 fastembed/ONNX 模型（`BAAI/bge-small-en-v1.5`，384 维，不引 torch），无 API key、无成本、无限流，门禁可完全离线跑；`gemini` 则用 Gemini `gemini-embedding-001`，通过原生 `batchEmbedContents` + `RETRIEVAL_QUERY` 调用，具备有界批处理、维度校验和异步重试。两条路都会对向量做 L2 归一化，改 provider/模型/维度都会让已录基线失效。这条 embedding 只给离线 answer-relevance 门禁打分；产品检索用的是另一条 embedding，不受影响。

质量门禁分成两条轨道：

- `replay` 读取已提交的 live 判官 fixture，不联网、不需要 API key，适合每次 push 的 CI。
- `live` 对每条 golden 问题只调用一次既有 `RAGPipeline.answer()`，固定生成答案和上下文，再让外部判官至少重复评估两次以测量 spread；必须同时显式开启 `RAGAS_ENABLED=true` 与 `RUN_RAGAS_EVAL=true`。

已有 live fixture 后运行离线门禁：

```bash
python -m eval.ragas_run replay
```

显式录制或刷新 live 基线：

```bash
python -m eval.ragas_run live --profile baseline --repetitions 3 --refresh-fixture
```

可分别录制能力 profile，并在保留判官方差的前提下比较：

```bash
python -m eval.ragas_run live --profile context_packing --repetitions 3 --fixture eval/fixtures/ragas-context-packing.jsonl --refresh-fixture
python -m eval.ragas_run compare --before eval/fixtures/ragas_verdicts.jsonl --after eval/fixtures/ragas-context-packing.jsonl --before-label baseline --after-label context-packing
```

门禁只覆盖消费生成答案的维度：Faithfulness、Answer Relevance 和负样本弃答/虚构进入 gate，Context Precision/Recall 不消费答案、只作为 reported-only 完整输出。生成提示按问题形状校准到“答案 + 最小支撑短语”的中间量——裸答案会让判官反推问题信息不足、过度脚手架又会漂离原问——配合版本化 statement 抽取，使 16 条正样本的 Faithfulness 与 Answer Relevance median 全部达标、负样本全部弃答。这份通过门禁的 capture 已提升为正式冻结 verdict fixture（schema `ragas-verdicts-v5`，pin 生成与抽取 prompt 版本、judge model、embedding 与 gating scope），`python -m eval.ragas_run replay` 离线自证门禁通过。

live 评估是受控数据出境面：正样本会把 question、生成 answer、检索 context 正文和人工 ground truth 发给外部判官；用 `gemini` provider 时 Answer Relevance 还会把文本发给 embedding endpoint，而本地 `bge` provider 不发起任何网络调用。fixture 只保存哈希和裁决，不保存评估正文。约 20 条人工样本上的四维分数只是带方差的判官估计，不是确定性事实、production 真值或质量保底；`temperature=0` 也不会消除供应商和模型漂移。

## 许可证

本仓库采用 [MIT License](LICENSE)。

## 代码规范

- 代码命名使用英文。
- 类和函数 docstring 使用英文。
- 关键逻辑注释使用英文。
- `README.md` 作为英文主文档。
- `README.zh-CN.md` 作为中文说明文档。
- 中文学习笔记单独放在 `docs/` 目录。

## 后续路线

1. 增加更多文档入库边界样例。
2. 增加分数阈值或拒答逻辑，改善 negative query。
3. 为现有观测钩子增加外部告警路由。
4. 增加 Qdrant 版本兼容等生产部署检查。
5. 增加内置认证或生产网关集成检查。
