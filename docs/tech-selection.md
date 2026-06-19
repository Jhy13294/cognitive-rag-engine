# 技术选型思考：企业级 RAG 知识库

本文记录当前项目为什么选择这些框架、模型、协议和工程边界。它不是安装手册，而是后续做架构复盘、替换组件和评估收益时的判断依据。

## 选型总原则

- **可替换**：embedding、vector store、reranker、query rewriter、cache 都通过抽象层接入，避免把供应商 SDK 写死在 RAG 主流程里。
- **可复现**：离线评估默认使用 `HashEmbeddingProvider` 和 deterministic fixture，先保证指标可逐位复现，再谈线上模型收益。
- **可降级**：网络 I/O、rerank、cache、query rewrite 失败时优先降级到已知稳定路径，而不是中断问答。
- **可观测**：关键阶段保留配置、来源、token、耗时、cache hit/miss 和错误分类，方便后续接入指标与审计。

## LLM API：DeepSeek 兼容聊天接口

当前聊天层通过 `api_client.py` 封装 DeepSeek 兼容接口。这样做的原因是：

- DeepSeek chat API 与 OpenAI chat completions 风格接近，便于复用通用请求/响应结构。
- 项目可以先稳定 prompt、retrieval 和 citation 逻辑，不把聊天模型选择耦合到检索漏斗。
- 聊天与 embedding 客户端已迁移到 `httpx.AsyncClient`，支持 async chat 和 streaming delta，服务层可以暴露真实 SSE。

工程边界：

- 上游 401、429、timeout、5xx 必须映射为结构化错误，HTTP 层不能靠消息子串猜状态。
- 同步 CLI 通过 `run_async_blocking()` 复用 async 实现，避免维护两套 retry 逻辑。
- 流式接口必须从 provider delta 逐步 yield，禁止先完整生成再切片。

## Embedding：OpenAI `text-embedding-3` + Hash 测试 Provider

生产 embedding 当前选择 OpenAI `text-embedding-3` 系列，测试使用 `HashEmbeddingProvider`。

选择 OpenAI embedding 的原因：

- `text-embedding-3` 系列支持 `dimensions` 参数，适合 Matryoshka 裁剪到 512 或 256 维，降低向量库成本。
- API 稳定、生态成熟，便于先把企业级 RAG 工程边界跑通。
- 与 `TokenCounter` 配合后可以做 token-aware batching，避免中文被旧 `len/4` 估算低估而超限。

保留 Hash provider 的原因：

- 不联网、不耗费 API 额度。
- 维度和输出确定，适合 CI、baseline 和重构防回归。
- 检索、重排、融合、查询改写和缓存基线的“逐位复现”都依赖它做稳定尺子。

边界提醒：

- Hash provider 不是语义模型，不能拿它的线上召回表现做生产结论。
- OpenAI provider 的 batch 构造必须走 `TokenCounter`，而不是字符数近似。
- query embedding 可以缓存，但 corpus embedding 不应在 query 热路径重算。

## Vector Store：Memory + Qdrant

项目保留两类向量库：

- `InMemoryVectorStore`：用于本地单测、离线 baseline 和快速验证。
- `QdrantVectorStore`：用于生产级持久化、跨进程 ingest/query 分离和 metadata filter。

选择 Qdrant 的原因：

- 原生向量检索能力成熟，支持 collection、payload、filter 和 scroll。
- 索引构建与查询分离要求 ingest 与 query 跨进程存活，进程内 dict 不能冒充持久化。
- 支持通过 payload 重建 BM25 corpus 和 parent store 所需文本，避免 query 时重 embed 语料。

边界提醒：

- `VectorRecord.id` 是 dense、BM25、RRF、rerank 之间对齐的核心。
- embedding model/dimension 必须在 ingest 与 query 之间一致；不一致要 raise，不能静默检索。
- memory store 只能证明职责分离或单元行为，不能证明持久化。

## Retrieval：Dense + BM25 + RRF

当前检索漏斗同时支持 dense、BM25 和 RRF 融合。

为什么引入 BM25：

- dense 检索擅长语义相似，但 exact name、编号、内部术语和长尾 query 经常需要词法匹配。
- 小型 golden set 中，Hash embedding 的 dense 路语义弱，BM25 可以清晰暴露稀疏召回的价值。
- BM25 不依赖外部网络，适合作为确定性 baseline 的一部分。

为什么选择 RRF：

- RRF 只依赖 rank，不强行比较 dense score 与 BM25 score 的数值尺度。
- 路径权重可配置，适合在 Hash dense 噪声较大时降低 dense 权重。
- 同一个 `ReciprocalRankFusion` 被 hybrid retrieval 和 multi-query retrieval 复用，减少融合器分叉。

边界提醒：

- BM25 输出 id 必须等于 `VectorRecord.id`，否则 RRF 会静默错配。
- Hybrid retrieval 的主评估必须在 reranker 关闭下测，否则 reranker 满分会掩盖融合贡献。

## Rerank：Deterministic + Cohere

Rerank 层分为离线确定性和生产 neural 两类。

- `DeterministicReranker`：读原文、零网络，用于 CI 门禁和冻结指标。
- `CohereReranker`：作为 neural provider，走 mock 和 gated 集成。

这个拆分避免把外部模型波动引入核心质量门禁。Rerank 层的重点是证明插入点、失败降级和指标口径正确，而不是把 CI 绑定到网络模型。

## Query Rewrite：Deterministic Fixture + Chat Provider

Query rewrite 是检索侧增强，允许指标变化，但必须防 query drift。

当前选择：

- `DeterministicQueryRewriter`：读取冻结 JSONL fixture，证明多查询 RRF 管线在已知优质改写下能工作。
- `ChatQueryRewriter`：生产态可注入聊天模型，失败时降级到原始 query。

红线：

- 原始 query 永远是 `q0`。
- `QUERY_REWRITE_WEIGHT_ORIGINAL >= QUERY_REWRITE_WEIGHT_VARIANT`。
- negative fixture 零覆盖时不能声称 multi-query-on-negative 安全。

## Service：FastAPI + SSE

选择 FastAPI 的原因：

- Pydantic 请求/响应模型天然适合做服务契约。
- 自动 OpenAPI 文档可作为 API 入口门禁。
- async endpoint 可以配合 `asyncio.to_thread` 保护事件循环，同时逐步把网络 I/O 迁到 async。

SSE 用于 `/query/stream`：

- 先发 `sources`，前端可以先展示引用。
- 再逐 token/delta 输出生成过程。
- L3 cache 命中时做 replay，并标注 `cached=true` 与 `stream_replay=true`，不冒充 live generation。

## Cache：Redis 三层缓存

Redis cache layer 使用 L1/L2/L3 三层设计：

- L1：query embedding，key 不含 `corpus_version`。
- L2：retrieval sources，key 含 Redis 持久化 `corpus_version`。
- L3：完整 `RAGResponse`，key 含 `corpus_version` 和生成配置。

选择 Redis 的原因：

- 跨进程、跨副本共享，比进程内 dict 更接近生产部署。
- `INCR` 可以作为 corpus version 的持久原子计数器。
- TTL 和 fail-open 策略能控制陈旧风险与服务可用性。
- `redis.asyncio` client 由 cache store 绑定到稳定后台 loop，既保留 async Redis I/O，又兼容同步 CLI、离线评估和服务 threadpool 调用。

边界提醒：

- Redis 挂了要降级为 miss，不能让 `/query` 500。
- 向量序列化用二进制 float64，不用 JSON float。
- 默认单测用 FakeRedis；live Redis smoke 和跨事件循环命中测试需要 `REDIS_URL` 显式开启。
- `redis.asyncio` 客户端不能作为普通进程单例在多个临时事件循环之间复用，store 需要持有稳定 loop 或改用同步 Redis client。
- 汇报缓存有效性时必须同时给出命中计数、上游调用计数和 Redis error 计数；冷/热指标不漂只能说明 fail-open 安全，不能单独证明缓存命中。

## 为什么暂未引入的组件

- **Redis 队列 / 后台任务队列**：`/ingest` 当前同步 offload，后台队列留到后续运维卡。
- **ACL/RBAC**：权限控制要在检索前过滤，不应混进缓存层。
- **Prometheus / OpenTelemetry**：审计与指标属于可观测性能力，当前只保留足够的统计入口。
- **Ragas**：生成质量评估属于后续质量评估能力；parent expansion 和 context packing 只能证明结构正确，不能提前宣称答案质量提升。
