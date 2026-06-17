# AI QA Assistant

这是一个面向企业级 RAG 知识库的早期 Python 项目。

当前项目重点是打好基础能力：

- DeepSeek 兼容的聊天 API 客户端
- 基于环境变量的配置管理
- 滚动文件日志
- TXT、Markdown、PDF、Word 文档加载
- 保守型文本清洗
- 面向向量入库的文本切分和元数据保留
- 生产级 Embedding 和向量库适配器
- 确定性检索评估和可选重排
- 共享词法检索基础能力
- Dense + BM25 混合检索和 RRF 融合
- Parent-Child 分块：子块检索、父块展开供给生成
- 可选上下文装填：整块纳入、确定性去重、引用连号和 token 预算
- Query Rewrite / Multi-Query：确定性改写 fixture 与跨 query RRF 融合

缓存、监控、API 服务层、权限控制等企业级能力尚未实现。

## 当前进度

当前阶段：可评估的检索漏斗，以及面向生成输入的上下文装填能力。

已完成：

- 基础 LLM API 调用流程
- 可复用 API 客户端和错误重试
- Document 数据模型和加载器抽象接口
- TXT 加载器
- 支持 Front Matter 的 Markdown 加载器
- 支持可选表格提取的 PDF 加载器
- Word `.docx` 加载器
- 文本清洗器
- 保留元数据的 RAG 文本切分器
- 统一文档加载入口
- 样例 fixture 和文档入库测试
- Embedding 抽象层
- 支持 dimensions、批处理、重试和 usage 日志的 OpenAI 生产级 Embedding Provider
- 共享 TokenCounter 抽象：支持可选 tiktoken 和离线确定性兜底
- 用于测试管线的本地确定性 Hash Embedding Provider
- 向量库抽象层
- 用于本地检索测试的内存向量库
- 支持批量写入、metadata 过滤、payload index 和重试机制的 Qdrant 生产级向量库适配器
- 向量库工厂和 CLI provider 选择
- 最小 RAG 管线：检索、上下文组装、LLM 生成、引用来源返回
- 确定性检索评估：JSONL golden set、HashEmbeddingProvider 基线、JSON/Markdown 报告
- Reranker 抽象层：确定性离线重排器和 Cohere neural reranker provider
- `RAGPipeline.retrieve` 可选接入重排：dense 召回、rerank、取 top-k，失败时降级回 dense 原序
- 共享 `lexical/` 核心：tokenization、IDF、BM25 和确定性词法评分
- BM25 稀疏检索器，输出与 dense 检索一致的 `VectorRecord.id`
- RRF 融合器，支持配置 dense/sparse 权重并稳定处理同分排序
- hybrid 对照评估：在关闭 reranker 的前提下并报 dense-only、bm25-only、fused 三路指标
- Parent-Child 分块器：只索引子块，父块通过 `parent_id` 键值取回
- 内存 `ParentStore`，用于本地确定性父块展开
- `RAGPipeline.retrieve` 支持 top-k 之后展开父块，并按父块折叠兄弟子块
- `ContextPacker` 只接入 build_prompt/answer 路径，支持整块装填、精确去重、可选近重复去重、引用连号和 token 预算
- OpenAI embedding batch 切分改为使用 TokenCounter，不再依赖旧的 `len/4` 估算
- QueryRewriter 抽象层：支持确定性 fixture 改写器和生产 Chat 改写器
- `RAGPipeline.retrieve` 支持 Multi-Query：原始 query + 改写变体，多路检索后用 RRF 融合，再进入既有 rerank、父块展开和上下文装填

尚未完成：

- 索引构建和查询命令拆分
- 缓存层
- 监控
- API 服务层
- 企业权限控制

## 目录结构

```text
.
├── api_client.py              # LLM API 客户端
├── config.py                  # 环境变量配置
├── logger.py                  # 日志工具
├── main.py                    # 命令行问答入口
├── rag_cli.py                 # RAG 应用命令行入口
├── requirements.txt           # Python 依赖
├── document_loader/
│   ├── base.py                # Document 模型和加载器接口
│   ├── chunking.py            # 文本切分
│   ├── loader.py              # 统一加载入口
│   ├── md_loader.py           # Markdown 加载器
│   ├── pdf_loader.py          # PDF 加载器
│   ├── txt_loader.py          # TXT 加载器
│   └── word_loader.py         # Word 加载器
├── embeddings/
│   ├── base.py                # Embedding 接口和向量工具
│   ├── openai_provider.py     # OpenAI 生产级 Embedding Provider
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
│   ├── fixtures/query_rewrites.jsonl # 确定性 query rewrite fixture
│   ├── baseline.py            # HashEmbeddingProvider 确定性基线
│   ├── metrics.py             # hit_rate、MRR、recall、negative 指标
│   ├── reporting.py           # JSON 和 Markdown 报告
│   ├── run.py                 # python -m eval.run 入口
│   ├── fixtures/              # 评估知识库样例
│   └── reports/               # 生成的评估报告
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
│   ├── chat.py                # 生产 Chat 改写器
│   └── factory.py             # Query rewriter 工厂
├── rag/
│   ├── context_packing.py     # top-k 后的上下文装填
│   └── pipeline.py            # 最小 RAG 管线
├── text_cleaner/
│   └── cleaner.py             # 文本清洗
├── tests/
│   ├── fixtures/              # TXT 和 Markdown 样例文档
│   ├── test_document_ingestion.py
│   ├── test_embeddings.py
│   ├── test_eval_metrics.py
│   ├── test_hybrid.py
│   ├── test_parent_child.py
│   ├── test_context_packing.py
│   ├── test_token_counter.py
│   ├── test_rerank.py
│   ├── test_qdrant_store_mock.py
│   ├── test_qdrant_store_integration.py
│   ├── test_vector_store.py
│   ├── test_rag_pipeline.py
│   └── test_rag_cli.py
└── docs/
    └── learning_notes.zh-CN.md
```

## 安装

创建并激活虚拟环境后，安装依赖：

```bash
pip install -r requirements.txt
```

创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions

EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL_NAME=text-embedding-3-small
EMBEDDING_DIMENSION=512

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

使用生产级 Embedding Provider 生成向量：

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

评估器只依赖 `retrieve(question, top_k)` 可调用对象，不调用聊天模型。默认基线使用 `HashEmbeddingProvider`，因此可以离线复现。

当前 T05 重排基线使用 `HashEmbeddingProvider` + `DeterministicReranker`：整体 MRR@3 从 `0.677083` 提升到 `1.000000`，long_tail MRR@3 从 `0.566667` 提升到 `1.000000`，recall@5 保持 `1.000000`。最新重排评估报告位于 `eval/reports/`。

当前 T06 hybrid 对照在关闭 reranker、使用 `HashEmbeddingProvider` 的条件下：dense-only MRR@3 为 `0.677083`，bm25-only MRR@3 为 `1.000000`，fused MRR@3 为 `1.000000`；fused 的 exact_name 和 long_tail MRR@3 也达到 `1.000000`，并保持报告逐字节可复现。最新 T06 报告位于 `eval/reports/`。

当前 T07 Parent-Child 模式只让子块进入检索和排序，父块只在最终 top-k 后展开供生成使用。评估命令默认关闭父块展开，只评子块 ranked list；因此检索指标的浮动属于“子块粒度变化”，不能记为父块展开带来的提升。生成侧连贯性和 Context Precision 的真账留到 T14 Ragas。

当前 T08 上下文装填只属于生成输入组装，不进入 `retrieve()`，不能被拿来声明 MRR 或 recall 提升。所有 context flag 关闭时，旧的字符制 `_build_context` 路径保持兼容；显式开启后，ContextPacker 会整块纳入或跳过，引用重新连号，精确重复默认由开关控制，可选近重复去重由独立 flag 守卫，并使用 TokenCounter 计算 token 预算。tiktoken 是可选依赖，缺失时自动使用确定性启发式兜底。对中文文本，真实 token 计数可能让 batch 变多、变小但合法；收益是避免超限请求，而不是“批数下降”。生成质量和连贯性结论仍延后到 T14 Ragas。

当前 T09 Multi-Query 是检索侧改动，因此 hit_rate、MRR、recall 是合法测量面。确定性 fixture 保证原始 query 始终作为 `q0`，只给 paraphrase 和 long_tail 样本增加冻结改写变体，多路检索后复用既有 RRF 融合，再进入既有 rerank、父块展开和上下文装填。离线 HashEmbeddingProvider 基线下，`python -m eval.run --multi-query` 将 paraphrase recall@3 从 `0.800000` 提升到 `1.000000`，long_tail recall@3 保持 `0.900000`，long_tail MRR@3 从 `0.566667` 提升到 `0.900000`。4 条 negative query 没有 fixture 改写，因此 multi-query 对它们整段旁路，结果与单路基线逐字节一致；multi-query 在 negative 上触发的风险没有被离线门禁覆盖，后续交给 roadmap 中的阈值 / abstain 工作处理。这些确定性 fixture 涨幅只是“给定已知优质改写时，RRF 融合管线能带来目标 paraphrase/long_tail 收益”的机制受控演示。线上 LLM 改写可能高于也可能低于这组数字，query drift 甚至可能跌破单路；这不是生产保底。

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
3. 增加缓存、可观测性和权限控制。
4. 增加 Qdrant 版本兼容与健康检查等生产部署门禁。
